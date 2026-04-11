use rustc_hash::{FxHashMap, FxHashSet};
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

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use num_traits::ToPrimitive;
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
        let mut ranges = Vec::with_capacity(self.packed_postings().len());
        for pack in self.packed_postings().iter() {
            ranges.push(pack.unpack());
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

        for (block_id, dot) in dots.into_iter().enumerate() {
            if heap.len() == k && dot < heap_factor * heap.peek().distance.distance() {
                continue;
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
        let dots: Vec<_> = dots
            .into_iter()
            .enumerate()
            .sorted_unstable_by(|&(_, a), &(_, b)| b.total_cmp(&a))
            .collect();

        for (block_id, dot) in dots {
            if heap.len() == k && dot < heap_factor * heap.peek().distance.distance() {
                continue;
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
            forward_index.prefetch_with_range(range);
        }

        for pack in packed_posting_block.iter() {
            let range = pack.unpack();

            if visited.insert(range.start) {
                let vector = forward_index.get_with_range(range.clone());
                let distance = evaluator.compute_distance(vector);
                heap.push(ScoredRange { distance, range });
            }
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

        // Batched prefetch + compute medoid distances (B=8).
        let distances: Vec<DotProduct> = {
            const BATCH: usize = 8;
            let mut dists = Vec::with_capacity(n_blocks);
            let mut batch_start = 0;
            while batch_start < n_blocks {
                let batch_end = (batch_start + BATCH).min(n_blocks);
                for i in batch_start..batch_end {
                    forward_index.prefetch_with_range(self.medoid_postings[i].unpack());
                }
                for i in batch_start..batch_end {
                    let medoid_range = self.medoid_postings[i].unpack();
                    dists.push(
                        evaluator.compute_distance(forward_index.get_with_range(medoid_range)),
                    );
                }
                batch_start = batch_end;
            }
            dists
        };

        // Scan: prune by medoid score, then evaluate non-medoid docs.
        for (block_id, medoid_dist) in distances.into_iter().enumerate() {
            if heap.len() == k
                && medoid_dist.distance() < heap_factor * heap.peek().distance.distance()
            {
                continue;
            }
            let medoid_range = self.medoid_postings[block_id].unpack();
            if visited.insert(medoid_range.start) {
                heap.push(ScoredRange {
                    distance: medoid_dist,
                    range: medoid_range,
                });
            }
            let packed_posting_block = &self.packed_postings()
                [self.block_offsets[block_id]..self.block_offsets[block_id + 1]];
            self.evaluate_posting_block(
                evaluator,
                packed_posting_block,
                heap,
                visited,
                forward_index,
            );
        }
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
            let _centroid_id = group[0].0;
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
    ) -> Vec<(ComponentFor<S>, f32)>
    where
        S: SeismicBuildDataset,
    {
        let mut hash = std::collections::HashMap::new();
        for &doc_id in block.iter() {
            let posting = dataset.get(doc_id as u64);
            let components = posting.components();
            let values = posting.values();
            for (&c, &v) in components.iter().zip(values) {
                let v = v.to_f32().unwrap();
                hash.entry(c)
                    .and_modify(|h| *h = if *h < v { v } else { *h })
                    .or_insert(v);
            }
        }

        hash.into_iter()
            .k_largest_by(n_components, |a, b| a.1.total_cmp(&b.1))
            .sorted_unstable_by_key(|&(id, _)| id)
            .collect()
    }

    fn energy_preserving_summary<S>(
        dataset: &S,
        block: &[usize],
        fraction: f32,
    ) -> Vec<(ComponentFor<S>, f32)>
    where
        S: SeismicBuildDataset,
    {
        let mut hash = std::collections::HashMap::new();
        for &doc_id in block.iter() {
            let posting = dataset.get(doc_id as u64);
            let components = posting.components();
            let values = posting.values();
            for (&c, &v) in components.iter().zip(values) {
                let v = v.to_f32().unwrap();
                hash.entry(c)
                    .and_modify(|h| *h = if *h < v { v } else { *h })
                    .or_insert(v);
            }
        }

        let mut components_values: Vec<_> = hash.into_iter().collect();

        components_values.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let total_sum: f32 = components_values
            .iter()
            .map(|(_, x)| x.to_f32().unwrap())
            .sum();

        let until = total_sum * fraction;
        let mut acc = 0_f32;
        components_values
            .into_iter()
            .take_while_inclusive(|(_, v)| {
                acc += v.to_f32().unwrap();
                acc < until
            })
            .sorted_unstable_by_key(|&(id, _)| id)
            .collect()
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
                            )
                        }

                        SummarizationStrategy::EnergyPreserving {
                            summary_energy: fraction,
                        } => Self::energy_preserving_summary(
                            dataset,
                            &posting_list[block_start..block_end],
                            fraction,
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
