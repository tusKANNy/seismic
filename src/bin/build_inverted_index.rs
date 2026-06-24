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
use vectorium::encoders::blocked_sparse::BlockedSparseEncoder;
use vectorium::encoders::dotpacking8_centroid4bit::DotPacking8Centroid4BitEncoder;
use vectorium::encoders::dotpacking8_fixedu8::DotPacking8FixedU8Encoder;
use vectorium::encoders::dotpacking8_scalar4bit::DotPacking8Scalar4BitEncoder;
use vectorium::encoders::dotpacking8_scalaru8::DotPacking8ScalarU8Encoder;
use vectorium::encoders::dotpacking8_f16::DotPacking8f16Encoder;
use vectorium::encoders::dotpacking8_u32_scalaru8::OptimisticDotPacking8U32ScalarU8Encoder;
use vectorium::encoders::dotpackingsparse_f16::DotPackingSparsef16Encoder;
use vectorium::encoders::dotpackingsparse_scalaru8::DotPackingSparseScalarU8Encoder;
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::dotvbyte_scalaru8::DotVByteScalarU8Encoder;
use vectorium::encoders::dotvbyte_u32_fixedu8::{
    DotVByteU32FixedU8Encoder, OptimisticDotVByteFixedU8Encoder,
};
use vectorium::encoders::dotvbyte_u32_scalaru8::{
    DotVByteU32ScalarU8Encoder, OptimisticDotVByteScalarU8Encoder,
};
use vectorium::{
    ComponentType, Dataset, DotProduct, IndexSerializer, PackedCentroidSparseQuantizer,
    PackedSparseDataset, PackedVariableBitUniformSparseQuantizer, SpaceUsage, read_seismic_format,
};

// ── Packed 4-bit quantizer hyperparameters ──────────────────────────────────
// vectorium trains the packed quantizers inside `convert_dataset_into` and reads
// their training hyperparameters *only* from process env vars (see vectorium's
// `quant_env`). Env vars are therefore the single channel we have to feed them, so
// we set them from the named constants below right before the conversion runs.
const QUANT_LOWER_PCT: f32 = 0.0; // no lower-percentile clipping
const QUANT_UPPER_PCT: f32 = 1.0; // no upper-percentile clipping
const QUANT_KMEANS_ITERS: usize = 5; // packed-centroid: per-component Lloyd iterations
// FIXME: forced 4-bit. The variable-bit uniform encoder supports nbits ∈ [1, 7], but
// we lock it to the specialized 4-bit operating point for now (SIMD-friendly
// interleaved nibble layout + dedicated dot-product kernel, byte-compatible with
// packed-centroid). Drop this constant / add a flag if other widths are needed later.
const QUANT_NBITS: u8 = 4;

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

    /// Value type: f16, bf16, f32, fixedu16, fixedu8, dotvbyte, dotvbyte-scalar, dotvbyte-optimistic, dotvbyte-optimistic-scalar, dotpacking8-fixedu8, dotpacking8-scalar, dotpacking8-sparse-scalar, dotpacking8-scalar4bit, dotpacking8-centroid4bit, dotpacking8-optimistic-scalar, dotpacking8-f16, dotpackingsparse-f16, packed-centroid, or packed-variable-uniform; see docs/RustUsage.md#using-the-rust-code for quantization choices.
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

fn build_blocked_sparse(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: blocked-sparse requires component-type 'u16'.");
        std::process::exit(1);
    }
    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<BlockedSparseEncoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotvbyte(args: &Args) {
    let time = Instant::now();
    match args.component_type.as_str() {
        "u16" => {
            let base_index = build_base_index::<u16>(args);
            let packed_index =
                base_index.convert_dataset_into::<PackedSparseDataset<DotVByteFixedU8Encoder>>();
            write_index(packed_index, args.output_file.as_ref().unwrap(), time);
        }
        "u32" => {
            let base_index = build_base_index::<u32>(args);
            let packed_index =
                base_index.convert_dataset_into::<PackedSparseDataset<DotVByteU32FixedU8Encoder>>();
            write_index(packed_index, args.output_file.as_ref().unwrap(), time);
        }
        _ => {
            eprintln!("Error: dotvbyte requires component-type 'u16' or 'u32'.");
            std::process::exit(1);
        }
    }
}

fn build_dotvbyte_scalar(args: &Args) {
    let time = Instant::now();
    match args.component_type.as_str() {
        "u16" => {
            let base_index = build_base_index::<u16>(args);
            let packed_index =
                base_index.convert_dataset_into::<PackedSparseDataset<DotVByteScalarU8Encoder>>();
            write_index(packed_index, args.output_file.as_ref().unwrap(), time);
        }
        "u32" => {
            let base_index = build_base_index::<u32>(args);
            let packed_index = base_index
                .convert_dataset_into::<PackedSparseDataset<DotVByteU32ScalarU8Encoder>>();
            write_index(packed_index, args.output_file.as_ref().unwrap(), time);
        }
        _ => {
            eprintln!("Error: dotvbyte-scalar requires component-type 'u16' or 'u32'.");
            std::process::exit(1);
        }
    }
}

fn build_dotvbyte_optimistic(args: &Args) {
    if args.component_type != "u32" {
        eprintln!("Error: dotvbyte-optimistic requires component-type 'u32'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u32>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<OptimisticDotVByteFixedU8Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotvbyte_optimistic_scalar(args: &Args) {
    if args.component_type != "u32" {
        eprintln!("Error: dotvbyte-optimistic-scalar requires component-type 'u32'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u32>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<OptimisticDotVByteScalarU8Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpacking8(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotpacking8 requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotPacking8FixedU8Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpacking8_fixedu8(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotpacking8-fixedu8 requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotPacking8FixedU8Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpacking8_scalar(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotpacking8-scalar requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotPacking8ScalarU8Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpacking8_sparse_scalar(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotpacking8-sparse-scalar requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotPackingSparseScalarU8Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpacking8_scalar4bit(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotpacking8-scalar4bit requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotPacking8Scalar4BitEncoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpacking8_centroid4bit(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotpacking8-centroid4bit requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotPacking8Centroid4BitEncoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpacking8_optimistic_scalar(args: &Args) {
    if args.component_type != "u32" {
        eprintln!("Error: dotpacking8-optimistic-scalar requires component-type 'u32'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u32>(args);
    let packed_index = base_index
        .convert_dataset_into::<PackedSparseDataset<OptimisticDotPacking8U32ScalarU8Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpacking8_f16(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotpacking8-f16 requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotPacking8f16Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_dotpackingsparse_f16(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: dotpackingsparse-f16 requires component-type 'u16'.");
        std::process::exit(1);
    }

    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<DotPackingSparsef16Encoder>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

/// Set the percentile-clipping env vars that vectorium's packed-quantizer
/// `ConvertFrom` reads at conversion time (shared by both packed value-types).
fn set_packed_quant_env_common() {
    // SAFETY: called at the top of a packed build function, before `build_base_index`
    // spawns the rayon-parallel build / conversion that reads these vars. No other
    // thread reads the environment at this point.
    unsafe {
        std::env::set_var("QUANT_LOWER_PCT", QUANT_LOWER_PCT.to_string());
        std::env::set_var("QUANT_UPPER_PCT", QUANT_UPPER_PCT.to_string());
    }
}

fn build_packed_centroid(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: packed-centroid requires component-type 'u16'.");
        std::process::exit(1);
    }
    set_packed_quant_env_common();
    // SAFETY: see set_packed_quant_env_common.
    unsafe {
        std::env::set_var("QUANT_KMEANS_ITERS", QUANT_KMEANS_ITERS.to_string());
    }
    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index =
        base_index.convert_dataset_into::<PackedSparseDataset<PackedCentroidSparseQuantizer>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn build_packed_variable_uniform(args: &Args) {
    if args.component_type != "u16" {
        eprintln!("Error: packed-variable-uniform requires component-type 'u16'.");
        std::process::exit(1);
    }
    set_packed_quant_env_common();
    // SAFETY: see set_packed_quant_env_common. QUANT_NBITS is forced to 4 (see the
    // const's FIXME) — we do not expose other bit widths from the CLI for now.
    unsafe {
        std::env::set_var("QUANT_NBITS", QUANT_NBITS.to_string());
    }
    let time = Instant::now();
    let base_index = build_base_index::<u16>(args);
    let packed_index = base_index
        .convert_dataset_into::<PackedSparseDataset<PackedVariableBitUniformSparseQuantizer>>();
    write_index(packed_index, args.output_file.as_ref().unwrap(), time);
}

fn main() {
    let args = Args::parse();

    if args.value_type == "blocked-sparse" {
        build_blocked_sparse(&args);
        return;
    }

    if args.value_type == "dotvbyte" {
        build_dotvbyte(&args);
        return;
    }

    if args.value_type == "dotvbyte-scalar" {
        build_dotvbyte_scalar(&args);
        return;
    }

    if args.value_type == "dotpacking8-fixedu8" {
        build_dotpacking8_fixedu8(&args);
        return;
    }

    if args.value_type == "dotpacking8-scalar" {
        build_dotpacking8_scalar(&args);
        return;
    }

    if args.value_type == "dotpacking8-sparse-scalar" {
        build_dotpacking8_sparse_scalar(&args);
        return;
    }

    if args.value_type == "dotpacking8-scalar4bit" {
        build_dotpacking8_scalar4bit(&args);
        return;
    }

    if args.value_type == "dotpacking8-centroid4bit" {
        build_dotpacking8_centroid4bit(&args);
        return;
    }

    if args.value_type == "dotvbyte-optimistic" {
        build_dotvbyte_optimistic(&args);
        return;
    }

    if args.value_type == "dotvbyte-optimistic-scalar" {
        build_dotvbyte_optimistic_scalar(&args);
        return;
    }

    if args.value_type == "dotpacking8-optimistic-scalar" {
        build_dotpacking8_optimistic_scalar(&args);
        return;
    }

    if args.value_type == "dotpacking8-f16" {
        build_dotpacking8_f16(&args);
        return;
    }

    if args.value_type == "dotpackingsparse-f16" {
        build_dotpackingsparse_f16(&args);
        return;
    }

    if args.value_type == "packed-centroid" {
        build_packed_centroid(&args);
        return;
    }

    if args.value_type == "packed-variable-uniform" {
        build_packed_variable_uniform(&args);
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
