use clap::Parser;
use num_traits::FromPrimitive;
use seismic::InvertedIndexBase;
use seismic::PlainInvertedIndex;
use seismic::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};
use seismic::with_dataset_type;
use serde::Serialize;
use serde::de::DeserializeOwned;

use std::hash::Hash;
use std::time::Instant;

use vectorium::dataset::ConvertFrom;
use vectorium::{
    ComponentType, Dataset, DotProduct, IndexSerializer, SpaceUsage, read_seismic_format,
};

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

// clap does not support enums with associated values; keep CLI-only types in the bin.
#[derive(clap::ValueEnum, Default, Debug, Clone)]
#[clap(rename_all = "kebab-case")]
enum SummarizationStrategyClap {
    #[default]
    EnergyPreserving,
    Medoid,
    FixedSize,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Source collection file (`documents.bin` style); see docs/RustUsage.md#using-the-rust-code.
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

    /// Summarization strategy: energy-preserving (default), medoid, or fixed-size.
    #[clap(long, value_parser, default_value = "energy-preserving")]
    summarization_strategy: SummarizationStrategyClap,

    /// Summary energy fraction preserved; see docs/RustUsage.md#using-the-rust-code (Executing Queries section).
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 0.5)]
    summary_energy: f32,

    /// Number of summary components for fixed-size summarization.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    n_components: usize,

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

    /// Component type (`u16` or `u32`); see docs/RustUsage.md#using-the-rust-code for sizing guidance.
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type: f32, f16, bf16, fixedu8, fixedu16, scalaru8, scalaru16, blocked-sparse, blocked-sparse-scalaru8, blocked-sparse-scalaru16, dotpacking8-fixedu8, dotpacking8-scalaru8, dotpacking8-scalaru16, dotpacking8-f16, or dotpacking8-scalar4bit (see `seismic::value_types` for the component types each supports); see docs/RustUsage.md#using-the-rust-code for quantization choices.
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

    let summarization = match args.summarization_strategy {
        SummarizationStrategyClap::EnergyPreserving => SummarizationStrategy::EnergyPreserving {
            summary_energy: args.summary_energy,
        },
        SummarizationStrategyClap::Medoid => SummarizationStrategy::Medoid,
        SummarizationStrategyClap::FixedSize => SummarizationStrategy::FixedSize {
            n_components: args.n_components,
        },
    };

    Configuration::default()
        .pruning_strategy(pruning)
        .blocking_strategy(BlockingStrategy::RandomKmeans {
            centroid_fraction: args.centroid_fraction,
            min_cluster_size: args.min_cluster_size,
            clustering_algorithm: clustering,
        })
        .summarization_strategy(summarization)
        .knn(knn_config)
}

fn build_base_index<C>(args: &Args) -> PlainInvertedIndex<C, f32>
where
    C: ComponentType
        + vectorium::ComponentType
        + SpaceUsage
        + Hash
        + FromPrimitive
        + Serialize
        + DeserializeOwned,
{
    let dataset = read_dataset::<C>(args);
    let config = build_config(args);
    println!("\nBuilding the index...");
    println!("{:?}", config);

    PlainInvertedIndex::<C, f32>::build(dataset, config)
}

fn read_dataset<C>(args: &Args) -> vectorium::PlainSparseDataset<C, f32, DotProduct>
where
    C: ComponentType + FromPrimitive,
{
    let dataset =
        read_seismic_format::<C, f32, DotProduct>(args.input_file.as_ref().unwrap()).unwrap();

    println!("Number of Vectors: {}", dataset.len());
    println!("Number of Dimensions: {}", dataset.input_dim());
    println!(
        "Avg number of components: {:.2}",
        dataset.nnz() as f32 / dataset.len() as f32
    );
    dataset
}

fn write_index<T>(index: T, output_file: &str, elapsed: Instant)
where
    T: IndexSerializer + serde::Serialize,
{
    let build_time = elapsed.elapsed().as_secs();
    println!("Time to build {} secs", build_time);

    let path = output_file.to_string() + ".index.seismic";
    println!("Saving ... {}", path);
    let save_start = Instant::now();
    if let Err(err) = index.save_index(path.as_str()) {
        eprintln!("Failed to save index to {}: {:?}", path, err);
    }
    let save_time = save_start.elapsed().as_secs();
    println!("Time to save {} secs", save_time);
    println!("Total time {} secs", build_time + save_time);
}

fn main() {
    let args = Args::parse();
    let time = Instant::now();
    let output_file = args.output_file.clone().unwrap();

    // `$C` is the component type, `$T` the dataset of the index; see `seismic::value_types`
    // for the meaning of the three routes.
    macro_rules! build {
        ($C:ty, $T:ty, plain) => {
            write_index(build_base_index::<$C>(&args), &output_file, time)
        };
        ($C:ty, $T:ty, quantize_first) => {{
            let dataset = <$T>::convert_from(read_dataset::<$C>(&args), ());
            let config = build_config(&args);
            println!("\nBuilding the index...");
            println!("{:?}", config);
            write_index(
                InvertedIndexBase::<$T>::build(dataset, config),
                &output_file,
                time,
            )
        }};
        ($C:ty, $T:ty, convert_after) => {
            write_index(
                build_base_index::<$C>(&args).convert_dataset_into::<$T>(),
                &output_file,
                time,
            )
        };
    }

    if let Err(msg) = with_dataset_type!(
        args.component_type.as_str(),
        args.value_type.as_str(),
        build
    ) {
        eprintln!("Error: {msg}");
        std::process::exit(1);
    }
}
