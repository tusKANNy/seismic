use seismic::SeismicIndex;
use seismic::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};
use vectorium::{DotProduct, IndexSerializer, ScalarSparseQuantizer, SparseDataset};

use half::f16;

use clap::Parser;
use std::time::Instant;

// clap does not support enums with associated values; keep CLI-only types in the bin.
#[derive(clap::ValueEnum, Default, Debug, Clone)]
#[clap(rename_all = "kebab-case")]
enum PruningStrategyClap {
    FixedSize,
    #[default]
    GlobalThreshold,
    CoiThreshold,
}

// clap does not support enums with associated values; keep CLI-only types in the bin.
#[derive(clap::ValueEnum, Default, Debug, Clone)]
#[clap(rename_all = "kebab-case")]
#[allow(clippy::enum_variant_names)]
enum ClusteringAlgorithmClap {
    RandomKmeans,
    RandomKmeansInvertedIndex,
    #[default]
    RandomKmeansInvertedIndexApprox,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Source collection file (`.jsonl` or `.tar.gz`); see docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    input_file: Option<String>,

    /// Output index base path; the binary appends `.index.seismic`. See docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    output_file: Option<String>,

    /// Number of postings to retain per list; tuning hints appear in docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 6000)]
    n_postings: usize,

    /// Block size used for fixed-size blocking; see docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    block_size: usize,

    /// Fraction of each posting list used to define k-means centroids; see docs/RustUsage.md#using-the-rust-code.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.1)]
    centroid_fraction: f32,

    /// Summary energy fraction preserved; see docs/RustUsage.md#using-the-rust-code (Executing Queries section).
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 0.5)]
    summary_energy: f32,

    /// Selects the clustering algorithm used to cluster postings in each posting list; see docs/RustUsage.md#using-the-rust-code.
    #[clap(long, value_parser)]
    clustering_algorithm: ClusteringAlgorithmClap,

    /// Choose the pruning strategy for posting lists; see docs/RustUsage.md#using-the-rust-code.
    #[clap(long, value_parser)]
    pruning_strategy: PruningStrategyClap,

    /// Pruning factor used by the random k-means blocking (see docs/RustUsage.md#using-the-rust-code).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.005)]
    kmeans_pruning_factor: f32,

    /// Number of top components retained while clustering with random k-means (see docs/RustUsage.md#using-the-rust-code).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 15)]
    kmeans_doc_cut: usize,

    /// Minimum cluster size allowed for random k-means blocking (see docs/RustUsage.md#using-the-rust-code).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 2)]
    min_cluster_size: usize,

    /// Fraction of each posting list kept by the coi-threshold pruning strategy (top `alpha * len` entries, capped at `n_postings`); see docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 0.15)]
    alpha: f32,

    /// Regulates the largest length of a posting list as a factor of `n_postings`; see docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 1.5)]
    max_fraction: f32,

    /// Number of neighbors stored per vector; see docs/RustUsage.md#using-the-rust-code for the accuracy impact.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    knn: usize,

    /// Path to a precomputed nearest-neighbor file (see docs/RustUsage.md#using-the-rust-code).
    #[clap(long, value_parser)]
    knn_path: Option<String>,

    /// Value type: f16, bf16, or f32; see docs/RustUsage.md#using-the-rust-code for quantization choices.
    #[clap(short, long, value_parser)]
    #[arg(default_value = "f16")]
    value_type: String,
}

fn build_config(args: &Args) -> Configuration {
    let knn_config = KnnConfiguration::new(args.knn, args.knn_path.clone());

    let clustering = match args.clustering_algorithm {
        ClusteringAlgorithmClap::RandomKmeansInvertedIndexApprox => {
            ClusteringAlgorithm::RandomKmeansInvertedIndexApprox {
                doc_cut: args.kmeans_doc_cut,
            }
        }
        ClusteringAlgorithmClap::RandomKmeansInvertedIndex => {
            ClusteringAlgorithm::RandomKmeansInvertedIndex {
                pruning_factor: args.kmeans_pruning_factor,
                doc_cut: args.kmeans_doc_cut,
            }
        }
        ClusteringAlgorithmClap::RandomKmeans => ClusteringAlgorithm::RandomKmeans {},
    };

    let pruning = match args.pruning_strategy {
        PruningStrategyClap::FixedSize => PruningStrategy::FixedSize {
            n_postings: args.n_postings,
        },
        PruningStrategyClap::GlobalThreshold => PruningStrategy::GlobalThreshold {
            n_postings: args.n_postings,
            max_fraction: args.max_fraction,
        },
        PruningStrategyClap::CoiThreshold => PruningStrategy::CoiThreshold {
            alpha: args.alpha,
            n_postings: args.n_postings,
        },
    };

    Configuration::default()
        .pruning_strategy(pruning)
        .blocking_strategy(BlockingStrategy::RandomKmeans {
            centroid_fraction: args.centroid_fraction,
            min_cluster_size: args.min_cluster_size,
            clustering_algorithm: clustering,
        })
        .summarization_strategy(SummarizationStrategy::EnergyPreserving {
            summary_energy: args.summary_energy,
        })
        .knn(knn_config)
}

pub fn main() {
    let args = Args::parse();

    let time = Instant::now();

    let config = build_config(&args);
    println!("\nBuilding the index...");
    println!("{:?}", config);

    let collection_path = args.input_file.unwrap();

    type EncoderF32 = ScalarSparseQuantizer<u16, f32, f32, DotProduct>;
    type DatasetF32 = SparseDataset<EncoderF32>;
    type EncoderF16 = ScalarSparseQuantizer<u16, f32, f16, DotProduct>;
    type DatasetF16 = SparseDataset<EncoderF16>;

    let index_f32 = SeismicIndex::<DatasetF32>::from_json(&collection_path, config, None, true);
    let index: SeismicIndex<DatasetF16> = index_f32.convert_dataset_into();

    let elapsed = time.elapsed();
    println!(
        "Time to build {} secs (before serializing)",
        elapsed.as_secs()
    );

    let path = args.output_file.unwrap() + ".index.seismic";
    println!("Saving ... {}", path);
    if let Err(err) = index.save_index(path.as_str()) {
        eprintln!("Failed to save index to {}: {:?}", path, err);
    }

    let elapsed = time.elapsed();
    println!("Time to build {} secs", elapsed.as_secs());
}
