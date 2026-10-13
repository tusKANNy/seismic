use serde::{Deserialize, Serialize};

/// This struct should contain every configuraion parameter for building the index
/// that doesn't need to be "managed" at query time.
/// Examples are the pruning strategy and the clustering strategy.
/// These can be chosen with a if at building time but there is no need to
/// make any choice at query time.
///
/// However, there are parameters that influence choices at query time.
/// To avoid branches or dynamic dispatching, this kind of parametrization are
/// selected with generic types.
/// An example is the quantization strategy. Based on the chosen
/// quantization strategy, we need to chose the right function to call while
/// computing the distance between vectors.
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    pub(crate) pruning: PruningStrategy,
    pub(crate) blocking: BlockingStrategy,
    pub(crate) summarization: SummarizationStrategy,
    pub(crate) knn: KnnConfiguration,
}

impl Configuration {
    pub fn pruning_strategy(mut self, pruning: PruningStrategy) -> Self {
        self.pruning = pruning;
        self
    }

    pub fn blocking_strategy(mut self, blocking: BlockingStrategy) -> Self {
        self.blocking = blocking;
        self
    }

    pub fn summarization_strategy(mut self, summarization: SummarizationStrategy) -> Self {
        self.summarization = summarization;
        self
    }

    pub fn knn(mut self, knn: KnnConfiguration) -> Self {
        self.knn = knn;
        self
    }
}

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
/// Represents the possible choices for the strategy used to prune the posting lists at build time.
pub enum PruningStrategy {
    FixedSize {
        n_postings: usize,
    },
    GlobalThreshold {
        n_postings: usize,
        max_fraction: f32,
    },
    CoiThreshold {
        alpha: f32,
        n_postings: usize,
    },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        Self::GlobalThreshold {
            n_postings: 3500,
            max_fraction: 1.5,
        }
    }
}

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
pub enum BlockingStrategy {
    FixedSize {
        block_size: usize,
    },
    RandomKmeans {
        centroid_fraction: f32,
        min_cluster_size: usize,
        clustering_algorithm: ClusteringAlgorithm,
    },
}

impl Default for BlockingStrategy {
    fn default() -> Self {
        BlockingStrategy::RandomKmeans {
            centroid_fraction: 0.1,
            min_cluster_size: 2,
            clustering_algorithm: ClusteringAlgorithm::default(),
        }
    }
}

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
pub enum SummarizationStrategy {
    FixedSize { n_components: usize },
    EnergyPreserving { summary_energy: f32 },
    Medoid,
}

impl Default for SummarizationStrategy {
    fn default() -> Self {
        Self::EnergyPreserving {
            summary_energy: 0.4,
        }
    }
}

#[derive(PartialEq, Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    RandomKmeans {},
    RandomKmeansInvertedIndex { pruning_factor: f32, doc_cut: usize },
    RandomKmeansInvertedIndexApprox { doc_cut: usize },
}

impl Default for ClusteringAlgorithm {
    fn default() -> Self {
        Self::RandomKmeansInvertedIndexApprox { doc_cut: 15 }
    }
}

#[derive(PartialEq, Default, Debug, Clone, Serialize, Deserialize)]
pub struct KnnConfiguration {
    pub(crate) nknn: usize,
    pub(crate) knn_path: Option<String>,
}

impl KnnConfiguration {
    pub fn new(nknn: usize, knn_path: Option<String>) -> Self {
        KnnConfiguration { nknn, knn_path }
    }
}
