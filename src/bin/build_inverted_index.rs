use clap::Parser;
use half::{bf16, f16};
use num_traits::FromPrimitive;
use seismic::FixedU8Q;
use seismic::FixedU16Q;
use seismic::PlainInvertedIndex;
use seismic::ScalarInvertedIndex;
use seismic::ScalarSparseDataset;
use seismic::configurations::{
    BlockingStrategy, ClusteringAlgorithm, Configuration, KnnConfiguration, PruningStrategy,
    SummarizationStrategy,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use std::hash::Hash;
use std::time::Instant;

use vectorium::dataset::ConvertFrom;
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::{
    ComponentType, Dataset, DotProduct, IndexSerializer, PackedSparseDataset, SpaceUsage,
    read_seismic_format,
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

    /// Regulates the fraction of L1 mass preserved by the COI pruning strategy; see docs/RustUsage.md#using-the-rust-code.
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

    /// Value type: f16, bf16, f32, fixedu16, fixedu8, or dotvbyte; see docs/RustUsage.md#using-the-rust-code for quantization choices.
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
    let dataset =
        read_seismic_format::<C, f32, DotProduct>(args.input_file.as_ref().unwrap()).unwrap();

    println!("Number of Vectors: {}", dataset.len());
    println!("Number of Dimensions: {}", dataset.input_dim());
    println!(
        "Avg number of components: {:.2}",
        dataset.nnz() as f32 / dataset.len() as f32
    );

    let config = build_config(args);
    println!("\nBuilding the index...");
    println!("{:?}", config);

    PlainInvertedIndex::<C, f32>::build(dataset, config)
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

fn build_for_component<C>(args: &Args)
where
    C: ComponentType
        + vectorium::ComponentType
        + SpaceUsage
        + Hash
        + FromPrimitive
        + Serialize
        + DeserializeOwned,
{
    let time = Instant::now();
    let dataset_f32 =
        read_seismic_format::<C, f32, DotProduct>(args.input_file.as_ref().unwrap()).unwrap();

    println!("Number of Vectors: {}", dataset_f32.len());
    println!("Number of Dimensions: {}", dataset_f32.input_dim());
    println!(
        "Avg number of components: {:.2}",
        dataset_f32.nnz() as f32 / dataset_f32.len() as f32
    );

    let config = build_config(args);
    println!("\nBuilding the index...");
    println!("{:?}", config);

    match args.value_type.as_str() {
        "f32" => write_index(
            PlainInvertedIndex::<C, f32>::build(dataset_f32, config),
            args.output_file.as_ref().unwrap(),
            time,
        ),
        "f16" => {
            let dataset = ScalarSparseDataset::<C, f32, f16, DotProduct>::convert_from(dataset_f32);
            write_index(
                ScalarInvertedIndex::<C, f32, f16>::build(dataset, config),
                args.output_file.as_ref().unwrap(),
                time,
            );
        }
        "bf16" => {
            let dataset =
                ScalarSparseDataset::<C, f32, bf16, DotProduct>::convert_from(dataset_f32);
            write_index(
                ScalarInvertedIndex::<C, f32, bf16>::build(dataset, config),
                args.output_file.as_ref().unwrap(),
                time,
            );
        }
        "fixedu16" => {
            let dataset =
                ScalarSparseDataset::<C, f32, FixedU16Q, DotProduct>::convert_from(dataset_f32);
            write_index(
                ScalarInvertedIndex::<C, f32, FixedU16Q>::build(dataset, config),
                args.output_file.as_ref().unwrap(),
                time,
            );
        }
        "fixedu8" => {
            write_index(
                ScalarInvertedIndex::<C, f32, FixedU8Q>::convert_dataset_from(
                    PlainInvertedIndex::<C, f32>::build(dataset_f32, config),
                ),
                args.output_file.as_ref().unwrap(),
                time,
            )
        }
        _ => {
            eprintln!("Error: value-type must be 'f16', 'bf16', 'f32', 'fixedu16', or 'fixedu8'");
            std::process::exit(1);
        }
    }
}

fn build_dotvbyte(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotvbyte requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotVByteFixedU8Encoder>>();

    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn main() {
    let args = Args::parse();

    if args.value_type == "dotvbyte" {
        build_dotvbyte(&args);
        return;
    }

    match args.component_type.as_str() {
        "u16" => build_for_component::<u16>(&args),
        "u32" => build_for_component::<u32>(&args),
        _ => {
            eprintln!("Error: component-type must be either 'u16' or 'u32'");
            std::process::exit(1);
        }
    }
}
