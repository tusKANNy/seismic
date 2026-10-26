use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;
use std::hash::Hash;

use crate::QuantizedSummary;
use crate::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, SummarizationStrategy,
};
use crate::index_traits::{
    ComponentFor, EncoderFor, SeismicBuildDataset, SeismicSearchDataset, ValueFor,
};
use crate::utils::{
    KHeap, do_random_kmeans_on_docids, do_random_kmeans_on_docids_ii_approx_dot_product,
    do_random_kmeans_on_docids_ii_dot_product,
};

use serde::{Deserialize, Serialize};

use num_traits::{AsPrimitive, ToPrimitive};
use vectorium::dataset::ScoredRange;
use vectorium::{
    ComponentType, DatasetGrowable, Distance, DotProduct, QueryEvaluator, SpaceUsage,
    SparseDataset, SparseDatasetGrowable, SparseVectorEncoder, SparseVectorView, VectorEncoder,
};

/// Instead of storing doc_ids we store their offsets in the forward_index and the lengths of the vectors
/// This allows us to save the random accesses that would be needed to access exactly these values from the
/// forward index. The values of each doc are packed into a single u64 in `packed_postings`.
/// We use 48 bits for the offset and 16 bits for the length. This choice limits the size of the dataset to be 1<<48.
/// We use the forward index to convert the offsets of the top-k back to the id of the corresponding documents.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Hash, Ord, PartialOrd)]
pub(crate) struct PackedPostingBlock {
    n: u64,
}

impl PackedPostingBlock {
    #[inline]
    pub fn pack(range: std::ops::Range<usize>) -> Self {
        let start = range.start as u64;
        assert!(
            start < (1u64 << 48),
            "range.start exceeds 48-bit packing limit"
        );
        let len = range.len();
        assert!(
            len <= u16::MAX as usize,
            "range length exceeds 16-bit packing limit"
        );
        Self {
            n: (start << 16) | (len as u64),
        }
    }

    #[inline]
    pub fn unpack(&self) -> std::ops::Range<usize> {
        let start = (self.n >> 16) as usize;
        let len = (self.n & (u16::MAX as u64)) as usize;
        start..(start + len)
    }
}

impl SpaceUsage for PackedPostingBlock {
    fn space_usage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PostingList<C: ComponentType> {
    packed_postings: Box<[PackedPostingBlock]>,
    block_offsets: Box<[usize]>,
    summaries: Option<QuantizedSummary<C>>,
    #[serde(default)]
    use_medoid_pruning: bool,
    #[serde(default)]
    medoid_postings: Box<[PackedPostingBlock]>,
}

impl<C: ComponentType + SpaceUsage> SpaceUsage for PostingList<C> {
    fn space_usage_bytes(&self) -> usize {
        SpaceUsage::space_usage_bytes(&self.packed_postings)
            + SpaceUsage::space_usage_bytes(&self.block_offsets)
            + self.summaries.as_ref().map_or(0, |s| s.space_usage_bytes())
            + SpaceUsage::space_usage_bytes(&self.medoid_postings)
    }
}

impl<C: ComponentType> PostingList<C> {
    #[inline]
    #[allow(clippy::borrowed_box)]
    pub(crate) fn packed_postings(&self) -> &Box<[PackedPostingBlock]> {
        &self.packed_postings
    }

    #[inline]
    pub(crate) fn packed_postings_mut(&mut self) -> &mut [PackedPostingBlock] {
        &mut self.packed_postings
    }

    #[inline]
    #[allow(clippy::borrowed_box)]
    pub(crate) fn block_offsets(&self) -> &Box<[usize]> {
        &self.block_offsets
    }

    #[inline]
    pub(crate) fn summaries(&self) -> &QuantizedSummary<C> {
        self.summaries
            .as_ref()
            .expect("summaries not available for medoid index")
    }

    #[inline]
    pub(crate) fn summaries_space_bytes(&self) -> usize
    where
        C: SpaceUsage,
    {
        self.summaries.as_ref().map_or(0, |s| s.space_usage_bytes())
    }

    #[inline]
    pub(crate) fn medoid_postings(&self) -> &[PackedPostingBlock] {
        &self.medoid_postings
    }

    #[inline]
    pub(crate) fn medoid_postings_mut(&mut self) -> &mut [PackedPostingBlock] {
        &mut self.medoid_postings
    }

    pub(crate) fn get_all_doc_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut ranges =
            Vec::with_capacity(self.packed_postings().len() + self.medoid_postings().len());

        if !self.use_medoid_pruning {
            for pack in self.packed_postings().iter() {
                ranges.push(pack.unpack());
            }
            return ranges;
        }

        // Medoid lists keep the medoid of each block apart from the other docs of the block:
        // emit it first, followed by the rest of its block.
        for (block_id, medoid) in self.medoid_postings().iter().enumerate() {
            ranges.push(medoid.unpack());
            let block = &self.packed_postings()
                [self.block_offsets[block_id]..self.block_offsets[block_id + 1]];
            for pack in block.iter() {
                ranges.push(pack.unpack());
            }
        }

        ranges
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search<'e, 'q, S>(
        &self,
        evaluator: &mut <EncoderFor<S> as VectorEncoder>::Evaluator<'e>,
        query: &SparseVectorView<'q, C, f32>,
        k: usize,
        heap_factor: f32,
        heap: &mut KHeap<ScoredRange<DotProduct>>,
        visited: &mut FxHashSet<usize>,
        forward_index: &'e S,
    ) where
        S: SeismicSearchDataset,
    {
        if self.use_medoid_pruning {
            return self.search_with_medoid(
                evaluator,
                k,
                heap_factor,
                heap,
                visited,
                forward_index,
            );
        }

        let dots = self.summaries().distances(query);

        if heap.len() == k {
            // The pruning threshold, taken once. While the blocks of this list are evaluated
            // `heap.peek()` can only improve and `heap_factor` is positive, so the live
            // threshold is never smaller than this one: a block dropped here would have been
            // dropped by the live test as well. Taking it once turns the scan into a plain
            // pass over a slice of `f32`, instead of a chain of loads from the heap that the
            // compiler cannot hoist because the heap is mutated inside the loop.
            let snapshot = heap_factor * heap.peek().distance.distance();

            // A tiny fraction of the blocks survives, so the scan works one chunk at a time:
            // when no element of a chunk clears the threshold the whole chunk is skipped.
            // The per-element check is written as `!(d < snapshot)` so that it is the exact
            // negation of the chunk reduction: the two tests agree on every score and the
            // filter is equivalent to the scalar `if dot < threshold { skip }` it replaces.
            // Survivors are visited in place, against the live threshold.
            const CHUNK: usize = 64;
            for (c, chunk) in dots.chunks(CHUNK).enumerate() {
                if chunk.iter().fold(true, |acc, &d| acc & (d < snapshot)) {
                    continue;
                }
                for (i, &d) in chunk.iter().enumerate() {
                    if !(d < snapshot) {
                        self.visit_block_if_promising(
                            c * CHUNK + i,
                            d,
                            evaluator,
                            k,
                            heap_factor,
                            heap,
                            visited,
                            forward_index,
                        );
                    }
                }
            }
        } else {
            // The heap is not full yet, so there is no threshold to apply: blocks are visited in
            // order and the live test inside `visit_block_if_promising` starts pruning as soon
            // as the heap fills up.
            for (block_id, dot) in dots.iter().copied().enumerate() {
                self.visit_block_if_promising(
                    block_id,
                    dot,
                    evaluator,
                    k,
                    heap_factor,
                    heap,
                    visited,
                    forward_index,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sort_and_search<'e, 'q, S>(
        &self,
        evaluator: &mut <EncoderFor<S> as VectorEncoder>::Evaluator<'e>,
        query: &SparseVectorView<'q, C, f32>,
        k: usize,
        heap_factor: f32,
        heap: &mut KHeap<ScoredRange<DotProduct>>,
        visited: &mut FxHashSet<usize>,
        forward_index: &'e S,
    ) where
        S: SeismicSearchDataset,
    {
        if self.use_medoid_pruning {
            return self.search_with_medoid(
                evaluator,
                k,
                heap_factor,
                heap,
                visited,
                forward_index,
            );
        }

        let dots = self.summaries().distances(query);
        // Blocks are visited in decreasing summary score, but a full sort is wasted work: the
        // heap-factor threshold lets a tiny fraction of them through, and after the first few
        // blocks the threshold barely moves. So only the HEAD best blocks are selected and
        // sorted (O(n)); once they have set the threshold, the rest is filtered against it in
        // one linear pass and only the survivors get sorted. The threshold never decreases, so
        // a block dropped here would have been skipped by the sorted scan as well, and the
        // survivors are visited in the same order: the result is the same.
        const HEAD: usize = 32;
        let mut order: Vec<(usize, f32)> = dots.into_iter().enumerate().collect();
        let n = order.len();
        let head = HEAD.min(n);
        let descending = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1);
        if 0 < head && head < n {
            order.select_nth_unstable_by(head - 1, descending);
        }
        order[..head].sort_unstable_by(descending);

        for &(block_id, dot) in &order[..head] {
            self.visit_block_if_promising(
                block_id,
                dot,
                evaluator,
                k,
                heap_factor,
                heap,
                visited,
                forward_index,
            );
        }

        let mut tail: Vec<(usize, f32)> = if heap.len() == k {
            let threshold = heap_factor * heap.peek().distance.distance();
            order[head..]
                .iter()
                .copied()
                .filter(|&(_, dot)| !(dot < threshold))
                .collect()
        } else {
            order[head..].to_vec()
        };
        tail.sort_unstable_by(descending);

        for (block_id, dot) in tail {
            self.visit_block_if_promising(
                block_id,
                dot,
                evaluator,
                k,
                heap_factor,
                heap,
                visited,
                forward_index,
            );
        }
    }

    /// Evaluates block `block_id` unless its summary score falls under the heap-factor
    /// threshold of the current top-k.
    ///
    /// The threshold is re-read from the heap here: the snapshot taken in `search` only
    /// pre-filters, and the k-th best may have risen since, so a block that passed the
    /// snapshot can still be skipped. With a non-full heap every block is evaluated.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn visit_block_if_promising<'e, S>(
        &self,
        block_id: usize,
        dot: f32,
        evaluator: &<EncoderFor<S> as VectorEncoder>::Evaluator<'e>,
        k: usize,
        heap_factor: f32,
        heap: &mut KHeap<ScoredRange<DotProduct>>,
        visited: &mut FxHashSet<usize>,
        forward_index: &'e S,
    ) where
        S: SeismicSearchDataset,
    {
        if heap.len() == k && dot < heap_factor * heap.peek().distance.distance() {
            return;
        }
        let block_offsets = self.block_offsets();
        let packed_posting_block =
            &self.packed_postings()[block_offsets[block_id]..block_offsets[block_id + 1]];

        self.evaluate_posting_block(
            evaluator,
            packed_posting_block,
            heap,
            visited,
            forward_index,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_posting_block<'e, S>(
        &self,
        evaluator: &<EncoderFor<S> as VectorEncoder>::Evaluator<'e>,
        packed_posting_block: &[PackedPostingBlock],
        heap: &mut KHeap<ScoredRange<DotProduct>>,
        visited: &mut FxHashSet<usize>,
        forward_index: &'e S,
    ) where
        S: SeismicSearchDataset,
    {
        for pack in packed_posting_block.iter() {
            let range = pack.unpack();
            if visited.contains(&range.start) {
                continue;
            }
            forward_index.prefetch_with_range(range, PREFETCH_MAX_BYTES);
        }

        // Unvisited documents are scored six at a time: the batched kernel walks the six
        // vectors in lockstep, so their cache misses overlap instead of being paid one after
        // the other, and the six accumulators are independent. Each accumulator sums in the
        // same order as the single-vector kernel, so the scores are identical.
        let mut buf: [std::ops::Range<usize>; 6] = Default::default();
        let mut count = 0;
        for pack in packed_posting_block.iter() {
            let range = pack.unpack();
            if !visited.insert(range.start) {
                continue;
            }
            buf[count] = range;
            count += 1;
            if count == 6 {
                let distances = evaluator.compute_distances_batch6([
                    forward_index.get_with_range(buf[0].clone()),
                    forward_index.get_with_range(buf[1].clone()),
                    forward_index.get_with_range(buf[2].clone()),
                    forward_index.get_with_range(buf[3].clone()),
                    forward_index.get_with_range(buf[4].clone()),
                    forward_index.get_with_range(buf[5].clone()),
                ]);
                for (range, distance) in buf.iter().zip(distances) {
                    heap.push(ScoredRange {
                        distance,
                        range: range.clone(),
                    });
                }
                count = 0;
            }
        }
        for range in buf.iter().take(count) {
            let vector = forward_index.get_with_range(range.clone());
            let distance = evaluator.compute_distance(vector);
            heap.push(ScoredRange {
                distance,
                range: range.clone(),
            });
        }
    }

    fn search_with_medoid<'e, S>(
        &self,
        evaluator: &mut <EncoderFor<S> as VectorEncoder>::Evaluator<'e>,
        k: usize,
        heap_factor: f32,
        heap: &mut KHeap<ScoredRange<DotProduct>>,
        visited: &mut FxHashSet<usize>,
        forward_index: &'e S,
    ) where
        S: SeismicSearchDataset,
    {
        let n_blocks = self.medoid_postings.len();
        if n_blocks == 0 {
            return;
        }

        // The medoid of every block is scored against the query: these scores play the role
        // that the summary scores have in `search`. Medoids are prefetched and scored six at a
        // time with the batched kernel, as the documents of a block are in
        // `evaluate_posting_block`: the scores are the ones of the single-vector kernel.
        let mut distances: Vec<DotProduct> = Vec::with_capacity(n_blocks);
        let (groups, rest) = self.medoid_postings.as_chunks::<6>();
        for group in groups {
            for medoid in group {
                forward_index.prefetch_with_range(medoid.unpack(), PREFETCH_MAX_BYTES);
            }
            distances.extend(evaluator.compute_distances_batch6(std::array::from_fn(|i| {
                forward_index.get_with_range(group[i].unpack())
            })));
        }
        for medoid in rest {
            forward_index.prefetch_with_range(medoid.unpack(), PREFETCH_MAX_BYTES);
        }
        for medoid in rest {
            distances
                .push(evaluator.compute_distance(forward_index.get_with_range(medoid.unpack())));
        }

        // Blocks are pruned by the score of their medoid, in the order they are stored. The scan
        // is the one of `search`: with a full heap the threshold is taken once and the scores
        // are filtered one chunk at a time, and the survivors are checked again against the
        // live threshold. See there for why the blocks that get evaluated are the same.
        if heap.len() == k {
            let snapshot = heap_factor * heap.peek().distance.distance();

            const CHUNK: usize = 64;
            for (c, chunk) in distances.chunks(CHUNK).enumerate() {
                if chunk
                    .iter()
                    .fold(true, |acc, d| acc & (d.distance() < snapshot))
                {
                    continue;
                }
                for (i, &medoid_dist) in chunk.iter().enumerate() {
                    if !(medoid_dist.distance() < snapshot) {
                        self.visit_medoid_block_if_promising(
                            c * CHUNK + i,
                            medoid_dist,
                            evaluator,
                            k,
                            heap_factor,
                            heap,
                            visited,
                            forward_index,
                        );
                    }
                }
            }
        } else {
            for (block_id, medoid_dist) in distances.into_iter().enumerate() {
                self.visit_medoid_block_if_promising(
                    block_id,
                    medoid_dist,
                    evaluator,
                    k,
                    heap_factor,
                    heap,
                    visited,
                    forward_index,
                );
            }
        }
    }

    /// Medoid counterpart of `visit_block_if_promising`: block `block_id` is evaluated unless
    /// the score of its medoid falls under the heap-factor threshold of the current top-k.
    /// The medoid is a document of the block stored apart from the others, so it enters the
    /// heap with the score it already has.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn visit_medoid_block_if_promising<'e, S>(
        &self,
        block_id: usize,
        medoid_dist: DotProduct,
        evaluator: &<EncoderFor<S> as VectorEncoder>::Evaluator<'e>,
        k: usize,
        heap_factor: f32,
        heap: &mut KHeap<ScoredRange<DotProduct>>,
        visited: &mut FxHashSet<usize>,
        forward_index: &'e S,
    ) where
        S: SeismicSearchDataset,
    {
        if heap.len() == k && medoid_dist.distance() < heap_factor * heap.peek().distance.distance()
        {
            return;
        }
        let medoid_range = self.medoid_postings[block_id].unpack();
        if visited.insert(medoid_range.start) {
            heap.push(ScoredRange {
                distance: medoid_dist,
                range: medoid_range,
            });
        }
        let packed_posting_block =
            &self.packed_postings()[self.block_offsets[block_id]..self.block_offsets[block_id + 1]];
        self.evaluate_posting_block(
            evaluator,
            packed_posting_block,
            heap,
            visited,
            forward_index,
        );
    }

    fn fixed_size_blocking(posting_list: &[usize], block_size: usize) -> Vec<usize> {
        let mut result: Vec<_> = (0..posting_list.len() / block_size)
            .map(|i| i * block_size)
            .collect();
        if result.last() != Some(&posting_list.len()) {
            result.push(posting_list.len());
        }
        result
    }

    fn blocking_with_random_kmeans<S>(
        posting_list: &mut [usize],
        centroid_fraction: f32,
        min_cluster_size: usize,
        dataset: &S,
        clustering_algorithm: ClusteringAlgorithm,
    ) -> Vec<usize>
    where
        S: SeismicBuildDataset,
    {
        if posting_list.is_empty() {
            return Vec::new();
        }

        let n_centroids = ((centroid_fraction * posting_list.len() as f32) as usize).max(1);

        assert!(
            n_centroids <= u16::MAX as usize,
            "In the current implementation the number of centroids cannot be greater than u16::MAX. This is due that the quantized summary uses u16 to store the centroids ids (aka summaries ids).\n Please, decrease centroid_fraction!"
        );

        let mut reordered_posting_list = Vec::<_>::with_capacity(posting_list.len());
        let mut block_offsets = Vec::<_>::with_capacity(n_centroids + 1);
        block_offsets.push(0);

        // Build k-means clusters on the posting list
        let mut clusters = match clustering_algorithm {
            ClusteringAlgorithm::RandomKmeans {} => {
                do_random_kmeans_on_docids(posting_list, n_centroids, dataset, min_cluster_size)
            }
            ClusteringAlgorithm::RandomKmeansInvertedIndex {
                pruning_factor,
                doc_cut,
            } => do_random_kmeans_on_docids_ii_dot_product(
                posting_list,
                n_centroids,
                dataset,
                min_cluster_size,
                pruning_factor,
                doc_cut,
            ),
            ClusteringAlgorithm::RandomKmeansInvertedIndexApprox { doc_cut } => {
                do_random_kmeans_on_docids_ii_approx_dot_product(
                    posting_list,
                    n_centroids,
                    dataset,
                    min_cluster_size,
                    doc_cut,
                )
            }
        };

        clusters.sort_unstable_by_key(|a| a.0);

        for group in
            clusters.chunk_by(|&(centroid_id_a, _doc_id_a), &(centroid_id_b, _doc_id_b)| {
                centroid_id_a == centroid_id_b
            })
        {
            for &(_centroid_id, doc_id) in group {
                reordered_posting_list.push(doc_id);
            }
            block_offsets.push(reordered_posting_list.len());
        }

        posting_list.copy_from_slice(&reordered_posting_list);

        if *block_offsets.last().unwrap_or(&0) != posting_list.len() {
            block_offsets.push(posting_list.len());
        }

        block_offsets
    }

    fn fixed_size_summary<S>(
        dataset: &S,
        block: &[usize],
        n_components: usize,
        scratch: &mut SummaryScratch<ComponentFor<S>>,
    ) -> Vec<(ComponentFor<S>, f32)>
    where
        S: SeismicBuildDataset,
        ComponentFor<S>: Hash,
    {
        if n_components == 0 {
            return Vec::new();
        }
        accumulate_block_max(scratch, dataset, block);
        let pairs = &mut scratch.pairs;
        if pairs.len() > n_components {
            pairs.select_nth_unstable_by(n_components - 1, |a, b| b.1.total_cmp(&a.1));
            pairs.truncate(n_components);
        }
        pairs.sort_unstable_by_key(|&(id, _)| id);
        pairs.clone()
    }

    fn energy_preserving_summary<S>(
        dataset: &S,
        block: &[usize],
        fraction: f32,
        scratch: &mut SummaryScratch<ComponentFor<S>>,
    ) -> Vec<(ComponentFor<S>, f32)>
    where
        S: SeismicBuildDataset,
        ComponentFor<S>: Hash,
    {
        accumulate_block_max(scratch, dataset, block);
        let pairs = &mut scratch.pairs;

        pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let total_sum: f32 = pairs.iter().map(|&(_, v)| v).sum();

        let until = total_sum * fraction;
        let mut acc = 0_f32;
        let mut kept = pairs.len();
        for (i, &(_, v)) in pairs.iter().enumerate() {
            acc += v;
            if acc >= until {
                kept = i + 1;
                break;
            }
        }
        pairs.truncate(kept);
        pairs.sort_unstable_by_key(|&(id, _)| id);
        pairs.clone()
    }
}

/// Above this dimension the dense per-component table of [`SummaryScratch`]
/// (two `u32`s per component, allocated once per posting list) is replaced by
/// a hash map — the same cutoff vectorium uses to densify sparse queries.
const DENSE_SCRATCH_MAX_DIM: usize = 1 << 20;

/// Bytes of a forward-index vector prefetched before it is scored. vectorium 0.11.1 leaves the cap
/// to the caller; 512 is what posting-block scans want (vectorium 0.11.0 hard-coded it).
const PREFETCH_MAX_BYTES: usize = 512;

/// Component → index-into-`pairs` lookup of [`SummaryScratch`].
enum ComponentSlots<C> {
    /// Dense table; `slots[c]` is the index into `pairs` of the entry
    /// accumulating component `c`'s max, valid iff `stamps[c] == epoch`
    /// (epoch-stamped so switching to the next block is O(1) instead of a
    /// clear).
    Dense {
        stamps: Vec<u32>,
        slots: Vec<u32>,
        epoch: u32,
    },
    /// Fallback for very high `input_dim`, where allocating the dense table
    /// for every posting list would dominate the build.
    Sparse(FxHashMap<C, u32>),
}

/// Reusable scratch for block summarization: a component→slot table replacing
/// the per-block `HashMap<C, f32>`, dense for small `input_dim` and hash-based
/// above [`DENSE_SCRATCH_MAX_DIM`]. Both variants fill `pairs` identically.
/// One instance per [`PostingList::build`] call, reused across all blocks of
/// the list.
struct SummaryScratch<C> {
    table: ComponentSlots<C>,
    /// Distinct (component, max value) pairs of the current block, in
    /// first-touch order.
    pairs: Vec<(C, f32)>,
}

impl<C> SummaryScratch<C> {
    fn new(input_dim: usize) -> Self {
        if input_dim < DENSE_SCRATCH_MAX_DIM {
            Self::dense(input_dim)
        } else {
            Self::sparse()
        }
    }

    fn dense(input_dim: usize) -> Self {
        Self {
            table: ComponentSlots::Dense {
                stamps: vec![0; input_dim],
                slots: vec![0; input_dim],
                epoch: 0,
            },
            pairs: Vec::new(),
        }
    }

    fn sparse() -> Self {
        Self {
            table: ComponentSlots::Sparse(FxHashMap::default()),
            pairs: Vec::new(),
        }
    }
}

/// Fill `scratch.pairs` with one (component, max value) entry per distinct
/// component appearing in `block`'s documents.
fn accumulate_block_max<S>(
    scratch: &mut SummaryScratch<ComponentFor<S>>,
    dataset: &S,
    block: &[usize],
) where
    S: SeismicBuildDataset,
    ComponentFor<S>: Hash,
{
    let SummaryScratch { table, pairs } = scratch;
    pairs.clear();
    match table {
        ComponentSlots::Dense {
            stamps,
            slots,
            epoch,
        } => {
            *epoch += 1;
            let epoch = *epoch;
            for &doc_id in block.iter() {
                let posting = dataset.get(doc_id as u64);
                let components = posting.components();
                let values = posting.values();
                for (&c, &v) in components.iter().zip(values) {
                    let v = v.to_f32().unwrap();
                    let idx: usize = c.as_();
                    if stamps[idx] != epoch {
                        stamps[idx] = epoch;
                        slots[idx] = pairs.len() as u32;
                        pairs.push((c, v));
                    } else {
                        let slot = &mut pairs[slots[idx] as usize].1;
                        if v > *slot {
                            *slot = v;
                        }
                    }
                }
            }
        }
        ComponentSlots::Sparse(map) => {
            map.clear();
            for &doc_id in block.iter() {
                let posting = dataset.get(doc_id as u64);
                let components = posting.components();
                let values = posting.values();
                for (&c, &v) in components.iter().zip(values) {
                    let v = v.to_f32().unwrap();
                    match map.entry(c) {
                        Entry::Vacant(e) => {
                            e.insert(pairs.len() as u32);
                            pairs.push((c, v));
                        }
                        Entry::Occupied(e) => {
                            let slot = &mut pairs[*e.get() as usize].1;
                            if v > *slot {
                                *slot = v;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<C> PostingList<C>
where
    C: ComponentType + std::hash::Hash,
{
    /// Find the medoid of a block: the document with the highest dot product with
    /// the block centroid (average vector).
    fn find_medoid<S>(dataset: &S, block: &[usize]) -> usize
    where
        S: SeismicBuildDataset,
    {
        if block.is_empty() {
            return 0;
        }
        if block.len() == 1 {
            return block[0];
        }

        let mut centroid: FxHashMap<ComponentFor<S>, f32> = FxHashMap::default();
        for &doc_id in block.iter() {
            let posting = dataset.get(doc_id as u64);
            for (&c, &v) in posting.components().iter().zip(posting.values()) {
                *centroid.entry(c).or_insert(0.0) += v.to_f32().unwrap();
            }
        }
        let block_size = block.len() as f32;
        for val in centroid.values_mut() {
            *val /= block_size;
        }

        let mut best_doc = block[0];
        let mut best_dot = 0.0f32;
        for &doc_id in block.iter() {
            let posting = dataset.get(doc_id as u64);
            let dot: f32 = posting
                .components()
                .iter()
                .zip(posting.values())
                .filter_map(|(&c, &v)| centroid.get(&c).map(|&cv| v.to_f32().unwrap() * cv))
                .sum();
            if dot > best_dot {
                best_dot = dot;
                best_doc = doc_id;
            }
        }
        best_doc
    }

    /// Reorder a block in-place so its medoid is at position 0.
    fn reorder_with_medoid_first<S>(dataset: &S, block: &mut [usize])
    where
        S: SeismicBuildDataset,
    {
        if block.len() <= 1 {
            return;
        }
        let medoid_doc = Self::find_medoid(dataset, block);
        if let Some(pos) = block.iter().position(|&x| x == medoid_doc) {
            block.swap(0, pos);
        }
    }

    pub(crate) fn build<S>(
        dataset: &S,
        postings: &[(ValueFor<S>, usize)],
        config: &Configuration,
    ) -> Self
    where
        S: SeismicBuildDataset,
        EncoderFor<S>: SparseVectorEncoder<OutputComponentType = C>,
    {
        let mut posting_list: Vec<_> = postings.iter().map(|(_, docid)| *docid).collect();

        let block_offsets = match config.blocking {
            BlockingStrategy::FixedSize { block_size } => {
                Self::fixed_size_blocking(&posting_list, block_size)
            }

            BlockingStrategy::RandomKmeans {
                centroid_fraction,
                min_cluster_size,
                clustering_algorithm,
            } => Self::blocking_with_random_kmeans(
                &mut posting_list,
                centroid_fraction,
                min_cluster_size,
                dataset,
                clustering_algorithm,
            ),
        };

        if matches!(config.summarization, SummarizationStrategy::Medoid) {
            // Reorder each block so the medoid is first.
            for &[block_start, block_end] in block_offsets.array_windows() {
                Self::reorder_with_medoid_first(dataset, &mut posting_list[block_start..block_end]);
            }

            let n_blocks = block_offsets.len().saturating_sub(1);
            let mut medoid_postings = Vec::with_capacity(n_blocks);

            // packed_postings: non-medoid docs only; block_offsets re-computed accordingly.
            let mut new_block_offsets = Vec::with_capacity(n_blocks + 1);
            new_block_offsets.push(0usize);
            let mut packed_postings =
                Vec::with_capacity(posting_list.len().saturating_sub(n_blocks));

            for i in 0..n_blocks {
                let block_start = block_offsets[i];
                let block_end = block_offsets[i + 1];
                let medoid_id = posting_list[block_start];
                medoid_postings.push(PackedPostingBlock::pack(
                    dataset.range_from_id(medoid_id as u64),
                ));
                for &doc_id in &posting_list[block_start + 1..block_end] {
                    packed_postings.push(PackedPostingBlock::pack(
                        dataset.range_from_id(doc_id as u64),
                    ));
                }
                new_block_offsets.push(packed_postings.len());
            }

            return Self {
                packed_postings: packed_postings.into_boxed_slice(),
                block_offsets: new_block_offsets.into_boxed_slice(),
                summaries: None,
                use_medoid_pruning: true,
                medoid_postings: medoid_postings.into_boxed_slice(),
            };
        }

        let quantizer = vectorium::PlainSparseQuantizer::<C, f32, vectorium::DotProduct>::new(
            dataset.input_dim(),
            dataset.input_dim(),
        );
        let mut summary = SparseDatasetGrowable::new(quantizer);
        let mut scratch = SummaryScratch::new(dataset.input_dim());
        for (components, values) in
            block_offsets
                .array_windows()
                .map(|&[block_start, block_end]| {
                    let summary_vec = match config.summarization {
                        SummarizationStrategy::FixedSize { n_components } => {
                            Self::fixed_size_summary(
                                dataset,
                                &posting_list[block_start..block_end],
                                n_components,
                                &mut scratch,
                            )
                        }

                        SummarizationStrategy::EnergyPreserving {
                            summary_energy: fraction,
                        } => Self::energy_preserving_summary(
                            dataset,
                            &posting_list[block_start..block_end],
                            fraction,
                            &mut scratch,
                        ),

                        SummarizationStrategy::Medoid => unreachable!(),
                    };
                    let (components, values): (Vec<C>, Vec<f32>) = summary_vec.into_iter().unzip();
                    (components, values)
                })
        {
            summary.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }

        let packed_postings: Vec<_> = posting_list
            .iter()
            .map(|&doc_id| PackedPostingBlock::pack(dataset.range_from_id(doc_id as u64)))
            .collect();

        Self {
            packed_postings: packed_postings.into_boxed_slice(),
            block_offsets: block_offsets.into_boxed_slice(),
            summaries: Some(QuantizedSummary::from(SparseDataset::from(summary))),
            use_medoid_pruning: false,
            medoid_postings: Box::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorium::{PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer};

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    // The dense and hash-based scratch variants must fill `pairs` identically:
    // same first-touch order, same per-component maxima. Running several
    // successive blocks through the same scratches exercises the epoch-bump
    // reuse on one side and the `map.clear()` reuse on the other.
    #[test]
    fn test_summary_scratch_sparse_matches_dense() {
        let dim = 30;
        let n_docs = 64;
        let mut state = 42u64;
        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::new(quantizer);
        for _ in 0..n_docs {
            let mut components: Vec<u16> = (0..8)
                .map(|_| (lcg(&mut state) % dim as u64) as u16)
                .collect();
            components.sort_unstable();
            components.dedup();
            // Quantized values so duplicate components across the block's
            // docs actually collide and max-update.
            let values: Vec<f32> = components
                .iter()
                .map(|_| 0.1 + (lcg(&mut state) % 5) as f32 * 0.2)
                .collect();
            growable.push(SparseVectorView::new(
                components.as_slice(),
                values.as_slice(),
            ));
        }
        let dataset: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

        let mut dense = SummaryScratch::dense(dim);
        let mut sparse = SummaryScratch::sparse();
        let doc_ids: Vec<usize> = (0..n_docs).collect();
        for block in doc_ids.chunks(7) {
            accumulate_block_max(&mut dense, &dataset, block);
            accumulate_block_max(&mut sparse, &dataset, block);
            assert!(!dense.pairs.is_empty());
            assert_eq!(dense.pairs, sparse.pairs);
        }
    }
}
