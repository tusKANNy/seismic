use std::collections::{BinaryHeap, HashSet};

use itertools::Itertools;
use num_traits::{AsPrimitive, ToPrimitive};
use rand::prelude::*;

use crate::index_traits::{EncoderFor, SeismicBuildDataset};
use vectorium::{
    ComponentType, Distance, DotProduct, QueryEvaluator, SparseVectorView, ValueType, VectorEncoder,
};

/// A max-heap that keeps the k smallest elements seen so far.
#[derive(Clone)]
pub struct KHeap<T> {
    bh: BinaryHeap<T>,
    k: usize,
}

impl<T: Ord> KHeap<T> {
    #[inline]
    /// Create a heap that retains the k smallest elements; panics if `k == 0`.
    pub fn new(k: usize) -> Self {
        assert!(k > 0);
        Self {
            bh: BinaryHeap::with_capacity(k),
            k,
        }
    }

    #[inline]
    /// Insert an item, keeping only the k smallest elements.
    pub fn push(&mut self, item: T) {
        if self.bh.len() < self.k {
            self.bh.push(item);
        } else {
            let mut max = self.bh.peek_mut().unwrap();
            if item < *max {
                *max = item;
            }
        }
    }

    #[inline]
    /// Return the number of elements currently stored.
    pub fn len(&self) -> usize {
        self.bh.len()
    }

    #[inline]
    /// Return whether the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    /// Return the current maximum among the retained elements; panics if empty.
    pub fn peek(&self) -> &T {
        self.bh.peek().unwrap()
    }

    #[inline]
    /// Consume the heap and return the retained elements in ascending order.
    pub fn into_sorted_vec(self) -> Vec<T> {
        self.bh.into_sorted_vec()
    }
}

pub(crate) fn quantize<T: ValueType>(values: &[T]) -> (f32, f32, Vec<u8>) {
    assert!(!values.is_empty());

    const MAX_QUANT: f32 = u8::MAX as f32;

    // Compute min and max values in the vector
    let (min, max) = values
        .iter()
        .map(|v| v.to_f32().unwrap())
        .minmax_by(|a, b| a.total_cmp(b))
        .into_option()
        .unwrap();

    // Quantization splits the range [min, max] into n_classes blocks of equal size (max-min)/n_clasess.
    // (Exponential quantization could be possible as well.)
    let quant = (max - min) / MAX_QUANT;
    let quantized_values = values
        .iter()
        .map(|&v| ((v.to_f32().unwrap() - min) / quant).round() as u8)
        .collect();

    (min, quant, quantized_values)
}

fn iter_components_values<'a, C, V>(
    vector: &'a SparseVectorView<'a, C, V>,
) -> impl Iterator<Item = (C, V)> + 'a
where
    C: ComponentType,
    V: ValueType,
{
    vector
        .components()
        .iter()
        .copied()
        .zip(vector.values().iter().copied())
}

/// CSR-layout inverted index over the centroid vectors, mirroring the flat
/// offsets + postings layout of `QuantizedSummary`: one dense offset per
/// component and a single contiguous postings array, instead of one heap
/// allocation per component. Scores are converted to `f32` once at build time
/// so the scoring loop reads 8-byte `(centroid_slot, score)` entries.
pub(crate) struct FastInvertedIndex {
    offsets: Box<[u32]>,
    postings: Box<[(u32, f32)]>,
}

impl FastInvertedIndex {
    fn new<S>(dataset: &S, centroid_doc_ids: &[usize]) -> Self
    where
        S: SeismicBuildDataset,
    {
        let dim = dataset.input_dim();

        // Counting sort: per-component counts, prefix-sum, then fill.
        let mut offsets = vec![0_u32; dim + 1];
        for &centroid_doc_id in centroid_doc_ids {
            let posting = dataset.get(centroid_doc_id as u64);
            for &c in posting.components() {
                offsets[c.as_() + 1] += 1;
            }
        }
        for i in 1..=dim {
            offsets[i] += offsets[i - 1];
        }

        let mut postings = vec![(0_u32, 0_f32); offsets[dim] as usize];
        let mut cursors = offsets[..dim].to_vec();
        // Filling in slot order keeps each component's slice sorted by
        // centroid slot, i.e., the same accumulation order as a per-component
        // push loop.
        for (centroid_slot, &centroid_doc_id) in centroid_doc_ids.iter().enumerate() {
            let posting = dataset.get(centroid_doc_id as u64);
            for (c, score) in iter_components_values(&posting) {
                let cursor = &mut cursors[c.as_()];
                postings[*cursor as usize] = (centroid_slot as u32, score.to_f32().unwrap());
                *cursor += 1;
            }
        }

        Self {
            offsets: offsets.into_boxed_slice(),
            postings: postings.into_boxed_slice(),
        }
    }

    /// Keep only the `max_list_len` postings with the largest scores in each
    /// component's list, preserving the slot order of the survivors.
    fn pruned(self, max_list_len: usize) -> Self {
        let Self { offsets, postings } = self;
        let dim = offsets.len() - 1;
        let mut postings = postings.into_vec();
        let mut new_offsets = vec![0_u32; dim + 1];
        let mut write = 0_usize;
        for c in 0..dim {
            let start = offsets[c] as usize;
            let end = offsets[c + 1] as usize;
            let len = end - start;
            if len > max_list_len {
                let list = &mut postings[start..end];
                list.select_nth_unstable_by(max_list_len - 1, |a, b| b.1.total_cmp(&a.1));
                list[..max_list_len].sort_unstable_by_key(|&(slot, _)| slot);
            }
            let keep = len.min(max_list_len);
            postings.copy_within(start..start + keep, write);
            write += keep;
            new_offsets[c + 1] = write as u32;
        }
        postings.truncate(write);
        Self {
            offsets: new_offsets.into_boxed_slice(),
            postings: postings.into_boxed_slice(),
        }
    }

    #[inline]
    fn postings(&self, component_id: usize) -> &[(u32, f32)] {
        &self.postings[self.offsets[component_id] as usize..self.offsets[component_id + 1] as usize]
    }
}

/// The dissolve pass every clustering driver runs between its two assignment
/// passes: group a first-pass assignment by centroid, keep the clusters larger
/// than `min_cluster_size`, and hand back the documents of the dissolved ones
/// so the caller can reassign them.
///
/// Returns `(kept_assignments, to_be_reassigned, removed_centroids)`.
/// `kept_assignments` is sorted, since `centroid_assignments` is sorted before
/// grouping and the groups are emitted in order.
fn dissolve_small_clusters(
    mut centroid_assignments: Vec<(usize, usize)>,
    n_docs: usize,
    min_cluster_size: usize,
) -> (Vec<(usize, usize)>, Vec<usize>, HashSet<usize>) {
    let mut to_be_reassigned = Vec::new(); // docids that belong to too small clusters
    let mut kept_assignments = Vec::with_capacity(n_docs);
    let mut removed_centroids = HashSet::new();

    centroid_assignments.sort_unstable();

    for group in centroid_assignments.chunk_by(
        // group by centroid_id
        |&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| centroid_id_a == centroid_id_b,
    ) {
        let centroid_id = group[0].0;
        if group.len() <= min_cluster_size {
            to_be_reassigned.extend(group.iter().map(|(_centroid_id, doc_id)| doc_id));
            removed_centroids.insert(centroid_id);
        } else {
            kept_assignments.extend(group.iter());
        }
    }

    assert_eq!(
        to_be_reassigned.len() + kept_assignments.len(),
        n_docs,
        "Final assignment size mismatch"
    );

    (kept_assignments, to_be_reassigned, removed_centroids)
}

fn compute_centroid_assignments_approx_dot_product<S>(
    doc_ids: &[usize],
    inverted_index: &FastInvertedIndex,
    dataset: &S,
    centroids_doc_ids: &[usize],
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let mut scores = vec![0_f32; centroids_doc_ids.len()];

    doc_ids
        .iter()
        .map(|&doc_id| {
            scores.iter_mut().for_each(|v| *v = 0_f32);
            let posting = dataset.get(doc_id as u64);
            let iter = iter_components_values(&posting).k_largest_by(doc_cut, |a, b| {
                a.1.to_f32().unwrap().total_cmp(&b.1.to_f32().unwrap())
            });
            for (component_id, value) in iter {
                let value = value.to_f32().unwrap();
                let postings = inverted_index.postings(component_id.as_());
                for &(centroid_slot, score) in postings {
                    scores[centroid_slot as usize] += score * value;
                }
            }

            let (&max_centroid_doc_id, _) = centroids_doc_ids
                .iter()
                .zip(scores.iter())
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((&centroids_doc_ids[0], &0.0));

            (max_centroid_doc_id, doc_id)
        })
        .collect()
}

/// Perform a random k-means clustering on a set of document ids.
/// The function returns a vector of pairs (cluster_id, doc_id) where cluster_id is the id of the cluster to which the document belongs.
/// The vector is sorted by cluster_id.
///
/// The function uses a simple pruned inverted index to speed up the computation and computes the
/// true dot product between the document and the centroids.
/// The parameter `doc_cut` specifies how many components of the document vector to consider while computing the dot product.
pub(crate) fn do_random_kmeans_on_docids_ii_approx_dot_product<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let seed = 1142;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_doc_ids: Vec<_> = doc_ids
        .choose_multiple(&mut rng, n_clusters)
        .copied()
        .collect();

    // Build an inverted index for the centroids
    let inverted_index = FastInvertedIndex::new(dataset, &centroid_doc_ids);

    let centroid_assignments = compute_centroid_assignments_approx_dot_product(
        doc_ids,
        &inverted_index,
        dataset,
        &centroid_doc_ids,
        doc_cut,
    );

    // Prune too small clusters and reassign the documents to the closest cluster
    let (mut final_assignments, to_be_reassigned, removed_centroids) =
        dissolve_small_clusters(centroid_assignments, doc_ids.len(), min_cluster_size);

    // The first pass dissolves the large majority of the clusters, so scoring
    // the reassigned documents against `inverted_index` would spend most of its
    // time on centroids that are no longer eligible. Score them against a
    // second index holding only the survivors instead: the `scores` array, the
    // posting traversal and the argmax all shrink by the same factor.
    //
    // Filtering preserves the order of `centroid_doc_ids`, so each surviving
    // centroid accumulates the same summands in the same order and `max_by`
    // (which keeps the last maximum) breaks ties identically: the assignments
    // are bit-for-bit the ones the full index would have produced.
    if !to_be_reassigned.is_empty() {
        let surviving: Vec<usize> = centroid_doc_ids
            .iter()
            .copied()
            .filter(|centroid_doc_id| !removed_centroids.contains(centroid_doc_id))
            .collect();
        let surviving_index = FastInvertedIndex::new(dataset, &surviving);

        if surviving.is_empty() {
            // Every centroid that drew a document was dissolved and none was
            // left untouched. The full-index argmax used to reduce an empty
            // iterator here and fall back to the first centroid; keep that.
            final_assignments.extend(
                to_be_reassigned
                    .iter()
                    .map(|&doc_id| (centroid_doc_ids[0], doc_id)),
            );
        } else {
            final_assignments.extend(compute_centroid_assignments_approx_dot_product(
                to_be_reassigned.as_slice(),
                &surviving_index,
                dataset,
                &surviving,
                doc_cut,
            ));
        }
    }

    assert_eq!(
        final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assignments.sort();

    final_assignments
}

fn compute_centroid_assignments_dot_product<S>(
    doc_ids: &[usize],
    inverted_index: &FastInvertedIndex,
    dataset: &S,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
    for<'a> <EncoderFor<S> as VectorEncoder>::EncodedVector<'a>: Send,
    for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>:
        QueryEvaluator<<EncoderFor<S> as VectorEncoder>::EncodedVector<'a>, Distance = DotProduct>,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());

    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }

        let doc_vec = dataset.get(doc_id as u64);
        let max_centroid_id = {
            let evaluator = dataset.encoder().vector_evaluator(doc_vec);
            let mut visited = to_avoid.clone();

            // Sort query terms by score and evaluate the posting list only for the top ones
            let components = doc_vec.components();
            let values = doc_vec.values();
            let top_components: Vec<_> = components
                .iter()
                .copied()
                .zip(values.iter().copied())
                .k_largest_by(doc_cut, |a, b| {
                    a.1.to_f32().unwrap().total_cmp(&b.1.to_f32().unwrap())
                })
                .map(|(component_id, _value)| component_id)
                .collect();

            let mut max_centroid_id = centroids[0];
            let mut max_dot = 0.0_f32;
            for component_id in top_components {
                for &(centroid_slot, _score) in inverted_index.postings(component_id.as_()) {
                    let centroid_id = centroids[centroid_slot as usize];
                    if !visited.insert(centroid_id) {
                        continue;
                    }
                    let centroid_vec = dataset.get(centroid_id as u64);
                    let dot = evaluator.compute_distance(centroid_vec).distance();
                    if dot > max_dot {
                        max_dot = dot;
                        max_centroid_id = centroid_id;
                    }
                }
            }

            max_centroid_id
        };

        centroid_assignments.push((max_centroid_id, doc_id));
    }

    centroid_assignments
}

/// Perform a random k-means clustering on a set of document ids.
/// The function returns a vector of pairs (cluster_id, doc_id) where cluster_id is the id of the cluster to which the document belongs.
/// The vector is sorted by cluster_id.
///
/// The function uses a simple pruned inverted index to speed up the computation and computes the
/// true dot product between the document and the centroids.
/// The parameter `pruning_factor` controls the size of the pruned inverted index.
/// The parameter `doc_cut` specifies how many components of the document vector to consider while computing the dot product.
pub(crate) fn do_random_kmeans_on_docids_ii_dot_product<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
    pruning_factor: f32,
    doc_cut: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    let pruned_list_size = 5.max((doc_ids.len() as f32 * pruning_factor) as usize);

    // Build a pruned inverted index for the centroids
    let inverted_index = FastInvertedIndex::new(dataset, &centroid_ids).pruned(pruned_list_size);

    let centroid_assignments = compute_centroid_assignments_dot_product(
        doc_ids,
        &inverted_index,
        dataset,
        &centroid_ids,
        &HashSet::new(),
        doc_cut,
    );

    // Prune too small clusters and reassign the documents to the closest cluster
    let (mut final_assignments, to_be_reassigned, removed_centroids) =
        dissolve_small_clusters(centroid_assignments, doc_ids.len(), min_cluster_size);

    let centroid_assignments = compute_centroid_assignments_dot_product(
        to_be_reassigned.as_slice(),
        &inverted_index,
        dataset,
        &centroid_ids,
        &removed_centroids,
        doc_cut,
    );

    final_assignments.extend(centroid_assignments);

    assert_eq!(
        final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assignments.sort_unstable();

    final_assignments
}

fn compute_centroid_assignments<S>(
    doc_ids: &[usize],
    dataset: &S,
    centroids: &[usize],
    to_avoid: &HashSet<usize>,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let mut centroid_assignments = Vec::with_capacity(doc_ids.len());
    let centroid_set: HashSet<usize> = centroids.iter().copied().collect();

    for &doc_id in doc_ids.iter() {
        if centroid_set.contains(&doc_id) && !to_avoid.contains(&doc_id) {
            centroid_assignments.push((doc_id, doc_id));
            continue;
        }
        let doc_vec = dataset.get(doc_id as u64);
        let centroid_max = {
            let evaluator = dataset.encoder().vector_evaluator(doc_vec);

            let mut centroid_max = centroids[0];
            let mut max_dot = 0.0_f32;
            for &centroid_id in centroids {
                let centroid_vec = dataset.get(centroid_id as u64);
                let dot = evaluator.compute_distance(centroid_vec).distance();
                if dot > max_dot {
                    max_dot = dot;
                    centroid_max = centroid_id;
                }
            }

            centroid_max
        };

        centroid_assignments.push((centroid_max, doc_id));
    }

    centroid_assignments
}

/// Run randomized k-means on document ids using exact dot products for assignment.
/// Returns `(cluster_id, doc_id)` pairs sorted by `cluster_id`.
pub(crate) fn do_random_kmeans_on_docids<S>(
    doc_ids: &[usize],
    n_clusters: usize,
    dataset: &S,
    min_cluster_size: usize,
) -> Vec<(usize, usize)>
where
    S: SeismicBuildDataset,
{
    let seed = 42; // You can use any u64 value as the seed
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid_ids = doc_ids
        .iter()
        .copied()
        .choose_multiple(&mut rng, n_clusters);

    let centroid_assignments =
        compute_centroid_assignments(doc_ids, dataset, &centroid_ids, &HashSet::new());

    // Prune too small clusters and reassign the documents to the closest cluster
    let (mut final_assignments, to_be_reassigned, removed_centroids) =
        dissolve_small_clusters(centroid_assignments, doc_ids.len(), min_cluster_size);

    let centroid_assignments = compute_centroid_assignments(
        to_be_reassigned.as_slice(),
        dataset,
        &centroid_ids,
        &removed_centroids,
    );

    final_assignments.extend(&centroid_assignments);

    assert_eq!(
        final_assignments.len(),
        doc_ids.len(),
        "Final assignment size mismatch"
    );

    final_assignments.sort();

    final_assignments
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use vectorium::{
        Dataset, DatasetGrowable, PlainSparseDataset, PlainSparseDatasetGrowable,
        PlainSparseQuantizer,
    };

    #[test]
    fn test_fast_inverted_index_matches_naive_build() {
        let seed = 42;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n_vecs = 200;
        let dim = 500;

        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(quantizer);
        for _ in 0..n_vecs {
            let nnz = rng.random_range(10..=50);
            let mut components: Vec<usize> = (0..dim).collect();
            components.shuffle(&mut rng);
            components.truncate(nnz);
            components.sort_unstable();
            let components: Vec<u16> = components.into_iter().map(|c| c as u16).collect();
            let values: Vec<f32> = (0..nnz).map(|_| rng.random::<f32>()).collect();
            growable.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }
        let dataset: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

        let centroid_doc_ids: Vec<usize> = (0..n_vecs).step_by(3).collect();
        let fast = FastInvertedIndex::new(&dataset, &centroid_doc_ids);

        let mut naive = vec![Vec::new(); dataset.input_dim()];
        for (centroid_slot, &centroid_doc_id) in centroid_doc_ids.iter().enumerate() {
            let posting = dataset.get(centroid_doc_id as u64);
            for (c, score) in iter_components_values(&posting) {
                naive[c as usize].push((centroid_slot as u32, score));
            }
        }

        for c in 0..dataset.input_dim() {
            assert_eq!(
                fast.postings(c),
                naive[c].as_slice(),
                "mismatch at component {}",
                c
            );
        }

        // Pruning keeps, per component, the `max_list_len` largest-score
        // postings in slot order. Random f32 scores make ties negligible, so
        // the naive top-k below is unambiguous.
        let max_list_len = 5;
        let pruned = FastInvertedIndex::new(&dataset, &centroid_doc_ids).pruned(max_list_len);
        for c in 0..dataset.input_dim() {
            let mut expected = naive[c].clone();
            expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            expected.truncate(max_list_len);
            expected.sort_unstable_by_key(|&(slot, _)| slot);
            assert_eq!(
                pruned.postings(c),
                expected.as_slice(),
                "pruned mismatch at component {}",
                c
            );
        }
    }

    /// The reference implementation this port reproduces, transcribed from the version of
    /// `InvertedIndex::search` that produced the submitted numbers. It sorts (component, value)
    /// pairs and returns the prefix length, which is what the port computes from values alone.
    fn reference_cut(query_values: &[f32], energy_fraction: f32) -> usize {
        let total_energy = query_values.iter().sum::<f32>();
        let mut sorted_tuples: Vec<(usize, f32)> =
            query_values.iter().copied().enumerate().collect();
        sorted_tuples.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut accumulated_energy = 0.0;
        let mut break_index = sorted_tuples.len();
        for (i, (_comp, val)) in sorted_tuples.iter().enumerate() {
            if (accumulated_energy + val) / total_energy >= energy_fraction {
                break_index = i + 1;
                break;
            }
            accumulated_energy += val;
        }
        break_index
    }

    fn cut(query_values: &[f32], energy_fraction: f32) -> usize {
        energy_preserving_cut(query_values, energy_fraction, &mut Vec::new())
    }

    #[test]
    fn takes_smallest_prefix_reaching_the_requested_mass() {
        // Uniform values: half the mass needs half the components.
        assert_eq!(cut(&[1.0, 1.0, 1.0, 1.0], 0.5), 2);
        // The full mass needs every component.
        assert_eq!(cut(&[1.0, 1.0, 1.0, 1.0], 1.0), 4);
        // A single dominant component already covers the requested fraction.
        assert_eq!(cut(&[0.1, 0.7, 0.2], 0.6), 1);
        assert_eq!(cut(&[0.1, 0.7, 0.2], 0.85), 2);
    }

    #[test]
    fn considers_components_in_decreasing_order_of_value() {
        // Input order must not matter: only the sorted mass profile does.
        assert_eq!(cut(&[0.1, 0.7, 0.2], 0.6), cut(&[0.7, 0.2, 0.1], 0.6));
        assert_eq!(cut(&[0.2, 0.1, 0.7], 0.6), cut(&[0.7, 0.2, 0.1], 0.6));
    }

    #[test]
    fn degenerate_queries_fall_back_to_the_whole_query() {
        assert_eq!(cut(&[], 0.5), 0);
        // A zero total makes the fraction undefined; the reference scans everything.
        assert_eq!(cut(&[0.0, 0.0, 0.0], 0.5), 3);
    }

    #[test]
    fn matches_the_reference_implementation() {
        let queries: [&[f32]; 6] = [
            &[1.0, 1.0, 1.0, 1.0],
            &[0.1, 0.7, 0.2],
            &[2.5, 0.01, 1.25, 0.4, 0.9, 0.05],
            &[0.3],
            &[],
            &[0.0, 0.0, 0.0],
        ];

        for query in queries {
            for step in 1..=10 {
                let energy_fraction = step as f32 / 10.0;
                assert_eq!(
                    cut(query, energy_fraction),
                    reference_cut(query, energy_fraction),
                    "mismatch for query {query:?} at energy {energy_fraction}"
                );
            }
        }
    }

    #[test]
    fn scratch_buffer_is_reusable_across_queries() {
        let mut buf = Vec::new();
        assert_eq!(energy_preserving_cut(&[0.1, 0.7, 0.2], 0.6, &mut buf), 1);
        // A shorter query after a longer one must not see stale values.
        assert_eq!(energy_preserving_cut(&[1.0, 1.0], 1.0, &mut buf), 2);
        assert_eq!(energy_preserving_cut(&[0.1, 0.7, 0.2], 0.6, &mut buf), 1);
    }
}

/// Number of query components preserving at least `energy_fraction` of the query's total mass.
///
/// Components are taken in decreasing order of value; the cut is the smallest prefix whose
/// cumulative mass reaches `energy_fraction` of the total. Since `search` already scans the
/// top-`query_cut` components by value, passing this count as `query_cut` reproduces the
/// energy-preserving query pruning exactly.
///
/// This lives in the library rather than in `perf_inverted_index` because kNN graph construction
/// searches each document as a query and must prune it with the *same* rule the query side uses:
/// two copies of this function would be free to drift, and a drift would be invisible — the graph
/// would simply be built at an operating point nobody measured.
///
/// `buf` is scratch space reused across queries so the per-query cost stays allocation-free.
pub fn energy_preserving_cut(query_values: &[f32], energy_fraction: f32, buf: &mut Vec<f32>) -> usize {
    let total_energy: f32 = query_values.iter().sum();

    buf.clear();
    buf.extend_from_slice(query_values);
    buf.sort_unstable_by(|a, b| b.total_cmp(a));

    let mut accumulated = 0.0;
    for (i, &value) in buf.iter().enumerate() {
        // A non-positive total makes this comparison false throughout, falling back to the
        // full query, which is what the reference implementation does.
        if (accumulated + value) / total_energy >= energy_fraction {
            return i + 1;
        }
        accumulated += value;
    }

    buf.len()
}
