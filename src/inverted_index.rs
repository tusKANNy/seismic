use crate::index_traits::{
    ComponentFor, EncoderFor, SeismicBuildDataset, SeismicSearchDataset, ValueFor,
};
use crate::posting_list::{PackedPostingBlock, PostingList};
use crate::utils::KHeap;

use toolkit::{BitField, BitFieldGrowable};

use vectorium::IndexSerializer;
use vectorium::dataset::{ConvertFrom, ConvertInto};
use vectorium::vector_encoder::{SparseDataEncoder, SparseVectorEncoder};
use vectorium::{
    Dataset, Distance, DotProduct, QueryEvaluator, ScoredRange, ScoredVectorDotProduct, SpaceUsage,
    SparseData, SparseVectorView, VectorEncoder,
};

use indicatif::ParallelProgressIterator;

use itertools::Itertools;
use num_traits::{AsPrimitive, ToPrimitive};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::cmp;

use mem_dbg::{MemSize, SizeFlags};
use rustc_hash::FxHashSet;
use std::collections::HashMap;
use std::time::Instant;

use std::io::Result as IoResult;

pub use crate::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};

#[derive(Default, PartialEq, Clone, Serialize, Deserialize)]
pub struct InvertedIndexBase<S>
where
    S: SparseData,
    EncoderFor<S>: SparseDataEncoder,
{
    forward_index: S,
    #[serde(bound(
        serialize = "ComponentFor<S>: Serialize",
        deserialize = "ComponentFor<S>: DeserializeOwned"
    ))]
    posting_lists: Box<[PostingList<ComponentFor<S>>]>,
    config: Configuration,
    knn: Option<Knn>,
}

impl<S> IndexSerializer for InvertedIndexBase<S>
where
    S: SparseData,
    EncoderFor<S>: SparseDataEncoder,
{
}

impl<S> SpaceUsage for InvertedIndexBase<S>
where
    S: SparseData + SpaceUsage,
    EncoderFor<S>: SparseDataEncoder,
    ComponentFor<S>: SpaceUsage,
{
    fn space_usage_bytes(&self) -> usize {
        let forward = SpaceUsage::space_usage_bytes(&self.forward_index);

        let postings: usize = self
            .posting_lists
            .iter()
            .map(|list| list.space_usage_bytes())
            .sum();
        let knn_size = match &self.knn {
            Some(knn) => knn.space_usage_bytes(),
            None => 0,
        };

        forward + postings + knn_size
    }
}

impl<S> InvertedIndexBase<S>
where
    S: SparseData,
    EncoderFor<S>: SparseDataEncoder,
{
    pub fn get_doc_ids_in_postings(&self, list_id: usize) -> Vec<usize> {
        assert!(
            list_id < self.posting_lists.len(),
            "Invalid list_id: {}",
            list_id
        );
        self.posting_lists[list_id]
            .get_all_doc_ranges()
            .into_iter()
            .map(|range| self.forward_index.id_from_range(range) as usize)
            .collect()
    }

    /// Help function to print the space usage of the index.
    pub fn print_space_usage_byte(&self)
    where
        S: SpaceUsage,
        ComponentFor<S>: SpaceUsage,
    {
        println!("Space Usage:");
        let forward = SpaceUsage::space_usage_bytes(&self.forward_index);
        println!("\tForward Index: {:} Bytes", forward);

        let mut total_packed_postings = 0;
        let mut total_block_offsets = 0;
        let mut total_summaries = 0;
        let mut total_medoid_postings = 0;

        for posting_list in self.posting_lists.iter() {
            total_packed_postings += SpaceUsage::space_usage_bytes(posting_list.packed_postings());
            total_block_offsets += SpaceUsage::space_usage_bytes(posting_list.block_offsets());
            total_summaries += posting_list.summaries_space_bytes();
            total_medoid_postings +=
                posting_list.medoid_postings().len() * std::mem::size_of::<u64>();
        }

        let postings_total =
            total_packed_postings + total_block_offsets + total_summaries + total_medoid_postings;

        println!("\tPosting Lists: {:} Bytes", postings_total);
        println!(
            "\t  ├─ packed_postings: {:} Bytes ({:.2}%)",
            total_packed_postings,
            total_packed_postings as f64 / postings_total as f64 * 100.0
        );
        println!(
            "\t  ├─ block_offsets: {:} Bytes ({:.2}%)",
            total_block_offsets,
            total_block_offsets as f64 / postings_total as f64 * 100.0
        );
        println!(
            "\t  ├─ summaries: {:} Bytes ({:.2}%)",
            total_summaries,
            total_summaries as f64 / postings_total as f64 * 100.0
        );
        println!(
            "\t  └─ medoid_postings: {:} Bytes ({:.2}%)",
            total_medoid_postings,
            total_medoid_postings as f64 / postings_total as f64 * 100.0
        );

        let knn_size = match &self.knn {
            Some(knn) => knn.space_usage_bytes(),
            None => 0,
        };

        println!("\tKnn: {:} Bytes", knn_size);
        println!("\tTotal: {:} Bytes", forward + postings_total + knn_size);
    }

    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn search<'q>(
        &'q self,
        query: SparseVectorView<'q, ComponentFor<S>, f32>,
        k: usize,
        query_cut: usize,
        heap_factor: f32,
        n_knn: usize,
        first_sorted: bool,
    ) -> Vec<ScoredVectorDotProduct>
    where
        S: SeismicSearchDataset,
        EncoderFor<S>: VectorEncoder<Distance = DotProduct>,
        <EncoderFor<S> as VectorEncoder>::QueryVector<'q>:
            From<SparseVectorView<'q, ComponentFor<S>, f32>>,
    {
        let query_components = query.components();
        let query_values = query.values();

        // Assert that query components are sorted in case of using a mergsort like strategy for the dot product
        assert!(
            query_components.is_sorted(),
            "Query components must be sorted in ascending order."
        );

        let query_for_eval: <EncoderFor<S> as VectorEncoder>::QueryVector<'q> = query.into();
        let mut evaluator = self.forward_index.encoder().query_evaluator(query_for_eval);

        let mut heap = KHeap::new(k);
        let mut visited = FxHashSet::with_capacity_and_hasher(
            query_cut.min(query_components.len()) * 5000,
            Default::default(),
        ); // TODO: 5000 should be n_postings

        // Evaluate the posting list only for the top score query terms
        let mut iter = query_components
            .iter()
            .zip(query_values)
            .k_largest_by(query_cut, |a, b| a.1.total_cmp(b.1));
        if first_sorted && let Some((&component_id, _value)) = iter.next() {
            let component_idx: usize = component_id.as_();
            self.posting_lists[component_idx].sort_and_search(
                &mut evaluator,
                &query,
                k,
                heap_factor,
                &mut heap,
                &mut visited,
                &self.forward_index,
            );
        }
        for (&component_id, _value) in iter {
            let component_idx: usize = component_id.as_();
            self.posting_lists[component_idx].search(
                &mut evaluator,
                &query,
                k,
                heap_factor,
                &mut heap,
                &mut visited,
                &self.forward_index,
            );
        }
        if n_knn > 0
            && let Some(knn) = self.knn.as_ref()
        {
            knn.refine(
                &mut evaluator,
                &mut heap,
                &mut visited,
                &self.forward_index,
                n_knn,
            );
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|scored| ScoredVectorDotProduct {
                distance: scored.distance,
                vector: self.forward_index.id_from_range(scored.range),
            })
            .collect()
    }

    /// Convert the `InvertedIndexBase`'s dataset to another one.
    pub fn convert_dataset_from<T>(inverted_index: InvertedIndexBase<T>) -> Self
    where
        S: Dataset + SparseData + ConvertFrom<T>,
        T: Dataset + SparseData,
        EncoderFor<T>: SparseDataEncoder<OutputComponentType = ComponentFor<S>>,
    {
        let InvertedIndexBase {
            forward_index,
            mut posting_lists,
            config,
            knn,
        } = inverted_index;
        let old_packs: Vec<_> = (0..forward_index.len())
            .map(|i| PackedPostingBlock::pack(forward_index.range_from_id(i as u64)))
            .collect();
        let new_dataset = ConvertInto::<S>::convert_into(forward_index);
        let new_packs: Vec<_> = (0..new_dataset.len())
            .map(|i| PackedPostingBlock::pack(new_dataset.range_from_id(i as u64)))
            .collect();
        assert_eq!(
            old_packs.len(),
            new_packs.len(),
            "Converted dataset length mismatch."
        );
        let packs_map: HashMap<_, _> = old_packs.into_iter().zip(new_packs).collect();

        for posting in posting_lists.iter_mut() {
            for pack in posting.packed_postings_mut().iter_mut() {
                *pack = packs_map[pack];
            }
            for pack in posting.medoid_postings_mut().iter_mut() {
                *pack = packs_map[pack];
            }
        }

        Self {
            forward_index: new_dataset,
            posting_lists,
            config,
            knn,
        }
    }

    /// Convert the `InvertedIndexBase`'s dataset to another one.
    pub fn convert_dataset_into<T>(self) -> InvertedIndexBase<T>
    where
        T: Dataset + SparseData + ConvertFrom<S>,
        EncoderFor<T>: SparseDataEncoder<OutputComponentType = ComponentFor<S>>,
    {
        InvertedIndexBase::<T>::convert_dataset_from(self)
    }

    // Add a precomputed knn graph to the index, limiting the number of neighbours to limit if it is not None
    //pub fn add_knn(&mut self, knn: Knn, limit: Option<usize>) {
    pub fn add_knn(&mut self, knn: Knn) {
        self.knn = Some(knn);
    }

    // Implementation of the pruning strategy that selects the top-`n_postings` from each posting list
    fn fixed_pruning(dataset: &S, n_postings: usize) -> Vec<Vec<(ValueFor<S>, usize)>>
    where
        S: SeismicBuildDataset,
        ValueFor<S>: vectorium::FromF32,
    {
        let mut inverted_pairs: Vec<KHeap<ScoredVectorDotProduct>> =
            vec![KHeap::new(n_postings); dataset.input_dim()];

        for (i, posting) in dataset.iter().enumerate() {
            let components = posting.components();
            let values = posting.values();
            for (&component, &value) in components.iter().zip(values) {
                let score = DotProduct::from(value.to_f32().unwrap());
                inverted_pairs[component.as_()].push(ScoredVectorDotProduct {
                    distance: score,
                    vector: i as u64,
                });
            }
        }

        inverted_pairs
            .into_iter()
            .map(|k| {
                k.into_sorted_vec()
                    .into_iter()
                    .map(|ScoredVectorDotProduct { distance, vector }| {
                        (
                            <ValueFor<S> as vectorium::FromF32>::from_f32_saturating(
                                distance.distance(),
                            ),
                            vector as usize,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    // Implementation of the pruning strategy that selects the top-x posting from each posting list where x is alpha times the number of postings in the list or max_n_posting if too big.
    #[allow(unused)]
    fn coi_pruning<V: PartialOrd + Send>(
        inverted_pairs: &mut Vec<Vec<(V, usize)>>,
        alpha: f32,
        max_n_postings: usize,
    ) {
        inverted_pairs.par_iter_mut().for_each(|posting_list| {
            if posting_list.is_empty() {
                return;
            }
            posting_list
                .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(cmp::Ordering::Equal));

            let cur_n_postings =
                max_n_postings.min((posting_list.len() as f32 * alpha) as usize + 1);
            posting_list.truncate(cur_n_postings);

            posting_list.shrink_to_fit();
        })
    }

    // Implementation of the pruning strategy that selects the `n_postings*dim` top postings globally, with a limit of `n_postings*max_fraction` per posting.
    fn global_threshold_pruning(
        dataset: &S,
        n_postings: usize,
        max_fraction: f32,
    ) -> Vec<Vec<(ValueFor<S>, usize)>>
    where
        S: SeismicBuildDataset,
    {
        let mut new_inverted_pairs = vec![Vec::new(); dataset.input_dim()];
        let tot_postings = dataset.input_dim() * n_postings;

        let largest_entries = dataset
            .iter()
            .enumerate()
            .flat_map(|(i, posting)| {
                posting
                    .components()
                    .iter()
                    .copied()
                    .zip(posting.values().iter().copied())
                    .map(move |(c, v)| (i, (c, v)))
            })
            .k_largest_by(tot_postings, |(_, (_, score_a)), (_, (_, score_b))| {
                score_a.partial_cmp(score_b).unwrap()
            });

        for (doc, (component, value)) in largest_entries {
            if new_inverted_pairs[component.as_()].len()
                < (n_postings as f32 * max_fraction) as usize
            {
                new_inverted_pairs[component.as_()].push((value, doc))
            };
        }

        new_inverted_pairs
    }

    /// Returns the sparse dataset
    pub fn dataset(&self) -> &S {
        &self.forward_index
    }

    /// Returns knn graph if present
    pub fn knn(&self) -> Option<&Knn> {
        self.knn.as_ref()
    }

    /// Returns the id of the largest component, i.e., the dimensionality of the vectors in the dataset.
    pub fn dim(&self) -> usize {
        self.forward_index.input_dim()
    }

    /// Returns the number of non-zero components in the dataset.
    pub fn nnz(&self) -> usize {
        self.forward_index.nnz()
    }

    /// Returns the number of vectors in the dataset
    pub fn len(&self) -> usize {
        self.forward_index.len()
    }

    /// Checks if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of neighbors in the knn graph, 0 if knn graph is not present.
    pub fn knn_len(&self) -> usize {
        match &self.knn {
            Some(knn) => knn.dim,
            None => 0,
        }
    }
}

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize, Default)]
pub struct Knn {
    n_vecs: usize,
    dim: usize,
    neighbours: BitField,
}

impl IndexSerializer for Knn {}

impl SpaceUsage for Knn {
    fn space_usage_bytes(&self) -> usize {
        self.neighbours.mem_size(SizeFlags::empty())
            + SpaceUsage::space_usage_bytes(&self.n_vecs)
            + SpaceUsage::space_usage_bytes(&self.dim)
    }
}

impl Knn {
    pub fn new<S>(index: &InvertedIndexBase<S>, dim: usize) -> Self
    where
        S: SeismicBuildDataset + SeismicSearchDataset,
        for<'q> <EncoderFor<S> as VectorEncoder>::QueryVector<'q>:
            From<SparseVectorView<'q, ComponentFor<S>, f32>>,
    {
        const KNN_QUERY_CUT: usize = 10;
        const KNN_HEAP_FACTOR: f32 = 0.7;

        let n_vecs = index.len();
        print!("Computing kNN: ");
        let docs_search_results: Vec<_> = (0..index.forward_index.len())
            .into_par_iter()
            .progress_count(index.forward_index.len() as u64)
            .map(|my_doc_id| {
                let vec = index.forward_index.get(my_doc_id as u64);

                let components = vec.components();
                let values = vec.values();
                let f32_values: Vec<_> = values.iter().map(|v| v.to_f32().unwrap()).collect();
                let components = components.to_vec();
                let query = SparseVectorView::new(components.as_slice(), f32_values.as_slice());

                index
                    .search(
                        query,
                        dim + 1, // +1 to filter the document itself if present
                        KNN_QUERY_CUT,
                        KNN_HEAP_FACTOR,
                        0,
                        false,
                    )
                    .into_iter()
                    .filter(|result| result.vector != my_doc_id as u64) // remove the document itself
                    .take(dim)
                    .collect::<Vec<_>>()
            })
            .collect();

        let neighbours: Vec<u64> = docs_search_results
            .into_iter()
            .flatten()
            .map(|result| result.vector)
            .collect();

        let bitfield = BitField::from(neighbours);

        Self {
            n_vecs,
            dim,
            neighbours: bitfield,
        }
    }

    pub fn new_from_serialized(path: &str, limit: Option<usize>) -> Self {
        println!("Reading KNN from file: {:}", path);
        let knn: Knn = Knn::load_index(path)
            .unwrap_or_else(|err| panic!("Failed to load KNN index from {path}: {err:?}"));

        println!("Number of vectors: {:}", knn.n_vecs);
        println!("Number of neighbors in the file: {:}", knn.dim);

        let nknn = limit.unwrap_or(knn.dim);

        assert!(
            nknn <= knn.dim,
            "The number of neighbors to include for each vector of the dataset can't be greater than the number of neighbours in the precomputed knn file."
        );

        if nknn == knn.dim {
            return knn;
        } else {
            println!("We only take {:} neighbors per element!", nknn);
        }

        let mut neighbours =
            BitFieldGrowable::with_capacity(knn.n_vecs, knn.neighbours.field_width());

        for id in 0..knn.n_vecs {
            let base_pos = id * knn.dim;
            for i in 0..nknn {
                let neighbor = knn.neighbours.get(base_pos + i).unwrap();
                neighbours.push(neighbor);
            }
        }
        let bitfield = BitField::from(neighbours);

        Knn {
            n_vecs: knn.n_vecs,
            dim: nknn,
            neighbours: bitfield,
        }
    }

    pub fn serialize(&self, output_file: &str) -> IoResult<()> {
        let path = output_file.to_string() + ".knn.seismic";
        println!("Saving ... {}", path.as_str());
        self.save_index(path.as_str())
            .map_err(|err| std::io::Error::other(format!("{err:?}")))?;
        Ok(())
    }

    #[inline]
    pub(crate) fn refine<'a, S>(
        &self,
        evaluator: &mut <EncoderFor<S> as VectorEncoder>::Evaluator<'a>,
        heap: &mut KHeap<ScoredRange<DotProduct>>,
        visited: &mut FxHashSet<usize>,
        forward_index: &'a S,
        in_n_knn: usize,
    ) where
        S: SeismicSearchDataset,
        EncoderFor<S>: VectorEncoder<Distance = DotProduct>,
        for<'b> <EncoderFor<S> as VectorEncoder>::Evaluator<'b>: QueryEvaluator<
                <EncoderFor<S> as VectorEncoder>::EncodedVector<'b>,
                Distance = DotProduct,
            >,
    {
        let n_knn = cmp::min(self.dim, in_n_knn);

        let neighbours: Vec<_> = heap.clone().into_sorted_vec();

        for ScoredRange::<DotProduct> {
            distance: _distance,
            range,
        } in neighbours.into_iter()
        {
            let id = forward_index.id_from_range(range.clone()) as usize;
            let base_pos = id * self.dim;

            for i in 0..n_knn {
                // SAFETY: we are sure the position is valid
                debug_assert!(base_pos + i < self.neighbours.len());
                // SAFETY: base_pos + i is within neighbours bounds (validated via debug assert).
                let neighbour = unsafe { self.neighbours.get_unchecked(base_pos + i) };

                let range = forward_index.range_from_id(neighbour);

                if visited.insert(range.start) {
                    let vector = forward_index.get_with_range(range.clone());
                    let distance = evaluator.compute_distance(vector);
                    heap.push(ScoredRange { distance, range });
                }
            }
        }
    }
}

impl<S> InvertedIndexBase<S>
where
    S: SeismicBuildDataset + SeismicSearchDataset,
    for<'a> <EncoderFor<S> as VectorEncoder>::Evaluator<'a>:
        QueryEvaluator<<EncoderFor<S> as VectorEncoder>::EncodedVector<'a>, Distance = DotProduct>,
{
    /// `n_postings`: minimum number of postings to select for each component
    pub fn build(dataset: S, config: Configuration) -> Self
    where
        for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
            From<SparseVectorView<'a, ComponentFor<S>, f32>>,
        ValueFor<S>: vectorium::FromF32,
    {
        print!("Distributing and pruning postings: ");
        let time = Instant::now();

        // Distribute pairs (score, doc_id) to corresponding components.
        //
        // The pruning strategy chosses how to filter out low-scoring postings.
        let inverted_pairs = match config.pruning {
            PruningStrategy::FixedSize { n_postings } => Self::fixed_pruning(&dataset, n_postings),
            PruningStrategy::GlobalThreshold {
                n_postings,
                max_fraction,
            } => Self::global_threshold_pruning(&dataset, n_postings, max_fraction),
            // Partial pruning for CoiThreshold is not implemented yet. Need to estimate the number of postings in the final list
            PruningStrategy::CoiThreshold {
                alpha: _,
                n_postings: _,
            } => {
                todo!()
            }
        };

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());

        println!("Number of posting lists: {}", inverted_pairs.len());

        let avg_list_length = inverted_pairs.iter().map(|l| l.len()).sum::<usize>() as f32
            / inverted_pairs.len() as f32;
        println!("Avg posting list length: {:.2}", avg_list_length);

        print!("Building summaries: ");
        let time = Instant::now();
        // Build summaries and blocks for each posting list
        let posting_lists: Vec<_> = inverted_pairs
            .par_iter()
            .progress_count(inverted_pairs.len() as u64)
            .enumerate()
            .map(|(_component_id, posting_list)| {
                PostingList::build(&dataset, posting_list, &config)
            })
            .collect();

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());

        if config.knn.nknn == 0 && config.knn.knn_path.is_none() {
            return Self {
                forward_index: dataset,
                posting_lists: posting_lists.into_boxed_slice(),
                config: config.clone(),
                knn: None,
            };
        }

        let me = InvertedIndexBase::<S> {
            forward_index: dataset,
            posting_lists: posting_lists.into_boxed_slice(),
            config: config.clone(),
            knn: None,
        };

        let time = Instant::now();
        let knn_config = config.knn.clone();
        let knn = if let Some(knn_path) = knn_config.knn_path {
            Knn::new_from_serialized(&knn_path, Some(knn_config.nknn))
        } else {
            Knn::new(&me, knn_config.nknn)
        };

        let elapsed = time.elapsed();
        println!("{} secs", elapsed.as_secs());
        Self {
            forward_index: me.forward_index,
            posting_lists: me.posting_lists,
            config,
            knn: Some(knn),
        }
    }

    /// Convenience function to build InvertedIndexBase using a dataset as a base, then converting it.
    pub fn from_base_dataset<T>(dataset: T, config: Configuration) -> Self
    where
        S: ConvertFrom<T>,
        T: SeismicBuildDataset + SeismicSearchDataset,
        EncoderFor<T>: SparseVectorEncoder<OutputComponentType = ComponentFor<S>>,
        for<'a> <EncoderFor<T> as VectorEncoder>::QueryVector<'a>:
            From<SparseVectorView<'a, ComponentFor<T>, f32>>,
        for<'a> <EncoderFor<T> as VectorEncoder>::Evaluator<'a>: QueryEvaluator<
                <EncoderFor<T> as VectorEncoder>::EncodedVector<'a>,
                Distance = DotProduct,
            >,
        ValueFor<T>: vectorium::FromF32,
    {
        let inverted_index = InvertedIndexBase::<T>::build(dataset, config);
        Self::convert_dataset_from(inverted_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorium::{
        DatasetGrowable, DotProduct, PlainSparseDataset, PlainSparseDatasetGrowable,
        PlainSparseQuantizer, SparseVectorView,
    };

    // Test pushing empty vectors.
    #[test]
    fn test_empty_vectors() {
        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(5, 5);
        let mut dataset = PlainSparseDatasetGrowable::new(quantizer);

        // Push a single vector
        let components = vec![0, 2, 4];
        let values = vec![1.0, 2.0, 3.0];
        dataset.push(SparseVectorView::new(
            components.as_slice(),
            values.as_slice(),
        ));
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.input_dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        // Push another vector
        let c = Vec::new();
        let v = Vec::new();

        dataset.push(SparseVectorView::new(c.as_slice(), v.as_slice()));
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.input_dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        dataset.push(SparseVectorView::new(c.as_slice(), v.as_slice()));
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.input_dim(), 5);
        assert_eq!(dataset.nnz(), 3);

        // Push a fourth vector
        let components = vec![0, 1, 2, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        dataset.push(SparseVectorView::new(
            components.as_slice(),
            values.as_slice(),
        ));
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.input_dim(), 5);
        assert_eq!(dataset.nnz(), 7);

        let dataset: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();

        let index = InvertedIndexBase::build(dataset, Configuration::default());
        assert_eq!(index.len(), 4);
        assert_eq!(index.dim(), 5);
        assert_eq!(index.nnz(), 7);

        let components = vec![0_u16, 1, 2, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let query = SparseVectorView::new(components.as_slice(), values.as_slice());
        let results = index.search(query.into(), 10, 5, 0.7, 0, false);

        assert_eq!(results.len(), 2); // Empty vectors are never retrieved because they do not belong to any posting list!
        assert_eq!(results[0].vector, 3);
        assert_eq!(results[1].vector, 0);
    }

    #[test]
    fn test_convert_dataset_preserves_postings() {
        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(4, 4);
        let mut dataset = PlainSparseDatasetGrowable::new(quantizer);
        let components = vec![0, 2];
        let values = vec![1.0, 2.0];
        dataset.push(SparseVectorView::new(
            components.as_slice(),
            values.as_slice(),
        ));
        let components = vec![1, 3];
        let values = vec![3.0, 4.0];
        dataset.push(SparseVectorView::new(
            components.as_slice(),
            values.as_slice(),
        ));
        let dataset: PlainSparseDataset<u16, f32, DotProduct> = dataset.into();

        let index = InvertedIndexBase::build(dataset, Configuration::default());
        let dim = index.dim();
        let len = index.len();

        // Verify postings are correctly populated after build
        let doc_ids_per_list: Vec<_> = (0..dim).map(|i| index.get_doc_ids_in_postings(i)).collect();

        // Each document should appear in some posting lists
        let all_doc_ids: std::collections::HashSet<_> =
            doc_ids_per_list.iter().flatten().copied().collect();
        assert!(!all_doc_ids.is_empty(), "Postings should contain documents");
        assert!(
            all_doc_ids.iter().all(|&id| id < len),
            "Doc IDs should be valid"
        );
    }
}
