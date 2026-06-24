use std::cmp;
use std::fs::File;
use std::hash::Hash;
use std::io::Write;
use std::time::Instant;

use clap::Parser;
use num_traits::FromPrimitive;

use seismic::InvertedIndexBase;
use seismic::index_traits::{ComponentFor, EncoderFor};
use vectorium::IndexSerializer;

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
    ComponentType, Dataset, Distance, DotProduct, PackedCentroidSparseQuantizer,
    PackedSparseDataset, PackedVariableBitUniformSparseQuantizer, SpaceUsage, SparseVectorView,
    VectorEncoder, read_seismic_format,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the serialized index file (see docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, value_parser)]
    index_file: Option<String>,

    /// Query file containing ground-truth vectors (docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, value_parser)]
    query_file: Option<String>,

    /// Output file with the ranked results (see docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// Number of queries to evaluate; see docs/RustUsage.md#using-the-rust-code for recommended scaling.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10000)]
    n_queries: usize,

    /// Number of top-k results to retrieve; tuning hints appear in docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// Number of runs to average; see docs/RustUsage.md#using-the-rust-code.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 1)]
    n_runs: usize,

    /// This parameter introduces an efficiency/accuracy trade-off: only the top `query_cut` components are considered (see docs/RustUsage.md#using-the-rust-code).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10)]
    query_cut: usize,

    /// Controls block skipping based on estimated dot products (see docs/RustUsage.md#using-the-rust-code for the accuracy impact).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0.7)]
    heap_factor: f32,

    /// Number of neighbors of top results to rescore; see docs/RustUsage.md#using-the-rust-code for the intended accuracy trade-off.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 0)]
    n_knn: usize,

    /// Whether to sort the first component by estimated dot products before search (see docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, action)]
    #[arg(default_value_t = false)]
    first_sorted: bool,

    /// Optional query energy filter (see docs/RustUsage.md#using-the-rust-code).
    #[clap(long, value_parser)]
    query_energy: Option<f32>,

    /// Component type (`u16` or `u32`); see docs/RustUsage.md#using-the-rust-code for guidance.
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type (`f16`, `bf16`, `f32`, `fixedu8`, `fixedu16`, `dotvbyte`, `dotvbyte-scalar`, `dotvbyte-optimistic`, `dotvbyte-optimistic-scalar`, `dotpacking8-fixedu8`, `dotpacking8-scalar`, `dotpacking8-sparse-scalar`, `dotpacking8-scalar4bit`, `dotpacking8-centroid4bit`, `dotpacking8-optimistic-scalar`, `dotpacking8-f16`, `dotpackingsparse-f16`, `packed-centroid`, or `packed-variable-uniform`); see docs/RustUsage.md#using-the-rust-code for recommendations.
    #[clap(short, long, value_parser)]
    #[arg(default_value = "f16")]
    value_type: String,
}

impl Args {
    pub fn component_type(&self) -> &str {
        &self.component_type
    }

    pub fn value_type(&self) -> &str {
        &self.value_type
    }
}

macro_rules! match_component_value {
    ($component:expr, $value:expr, $macro_name:ident) => {
        use half::{bf16, f16};
        use seismic::{FixedU16Q, FixedU8Q};

        match ($component, $value) {
            ("u16", "f16") => {
                $macro_name!(u16, f16);
            }
            ("u16", "bf16") => {
                $macro_name!(u16, bf16);
            }
            ("u16", "f32") => {
                $macro_name!(u16, f32);
            }
            ("u16", "fixedu8") => {
                $macro_name!(u16, FixedU8Q);
            }
            ("u16", "fixedu16") => {
                $macro_name!(u16, FixedU16Q);
            }
            ("u32", "f16") => {
                $macro_name!(u32, f16);
            }
            ("u32", "bf16") => {
                $macro_name!(u32, bf16);
            }
            ("u32", "f32") => {
                $macro_name!(u32, f32);
            }
            ("u32", "fixedu8") => {
                $macro_name!(u32, FixedU8Q);
            }
            ("u32", "fixedu16") => {
                $macro_name!(u32, FixedU16Q);
            }
        _ => {
            eprintln!(
                "Error: component-type must be either 'u16' or 'u32', value-type must be 'f16', 'bf16', 'f32', 'fixedu16', 'fixedu8', 'dotvbyte', 'dotvbyte-scalar', 'dotvbyte-optimistic', 'dotvbyte-optimistic-scalar', 'dotpacking8-fixedu8', 'dotpacking8-scalar', 'dotpacking8-sparse-scalar', 'dotpacking8-scalar4bit', 'dotpacking8-centroid4bit', 'dotpacking8-optimistic-scalar', 'dotpacking8-f16', 'dotpackingsparse-f16', 'packed-centroid', or 'packed-variable-uniform'"
            );
            std::process::exit(1);
        }
        }
    };
}

pub fn run_performance_test_generic<S>(args: Args)
where
    S: seismic::SeismicSearchDataset + SpaceUsage + serde::Serialize + serde::de::DeserializeOwned,
    for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
        From<SparseVectorView<'a, ComponentFor<S>, f32>>,
    ComponentFor<S>: ComponentType
        + FromPrimitive
        + SpaceUsage
        + Hash
        + serde::Serialize
        + serde::de::DeserializeOwned,
{
    let index_path = args.index_file;
    let query_cut = args.query_cut;
    let heap_factor = args.heap_factor;
    let n_runs = args.n_runs;

    let nknn = args.n_knn;

    let inverted_index: InvertedIndexBase<S> =
        InvertedIndexBase::load_index(index_path.unwrap().as_str())
            .unwrap_or_else(|err| panic!("Failed to load index: {err:?}"));

    let queries =
        read_seismic_format::<ComponentFor<S>, f32, DotProduct>(&args.query_file.unwrap()).unwrap();

    let n_queries = cmp::min(args.n_queries, queries.len());

    println!("Searching for top-{} results", args.k);
    println!("Number of evaluated queries: {n_queries}");
    println!(
        "Avg number of non-zero components: {}",
        queries.nnz() / queries.len()
    );

    println!("Number of documents: {}", inverted_index.len());
    println!(
        "Avg number of non-zero components: {}",
        inverted_index.nnz() / inverted_index.len()
    );

    let mut results = Vec::with_capacity(n_queries);

    let time = Instant::now();
    for _ in 0..n_runs {
        results.clear();

        for (query_id, components_values) in queries.iter().take(n_queries).enumerate() {
            let query_view =
                SparseVectorView::new(components_values.components(), components_values.values());
            let query = query_view;
            let cur_results = inverted_index.search(
                query,
                args.k,
                query_cut,
                heap_factor,
                nknn,
                args.first_sorted,
            );

            if cur_results.len() < args.k {
                println!(
                    "FAIL! The query {query_id} has only {} results.",
                    cur_results.len()
                );
            }

            results.push(cur_results);
        }
    }
    let elapsed = time.elapsed();
    println!(
        "Time {} microsecs per query",
        elapsed.as_micros() / (n_runs * n_queries) as u128
    );
    eprintln!("{}", elapsed.as_micros() / (n_runs * n_queries) as u128);

    inverted_index.print_space_usage_byte();
    // Writes results to a file in a parsable format
    let output_path = args.output_path.unwrap();
    let mut output_file = File::create(output_path).unwrap();

    for (query_id, result) in results.iter().enumerate() {
        // Writes results to a file in a parsable format
        for (idx, scored) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{}\t{}\t{}",
                scored.vector,
                idx + 1,
                scored.distance.distance(),
            )
            .unwrap();
        }
    }
}

pub fn main() {
    let args = Args::parse();

    if args.value_type() == "blocked-sparse" {
        if args.component_type() != "u16" {
            eprintln!("Error: blocked-sparse requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<BlockedSparseEncoder>>(args);
        return;
    }

    if args.value_type() == "dotvbyte" {
        match args.component_type() {
            "u16" => {
                run_performance_test_generic::<PackedSparseDataset<DotVByteFixedU8Encoder>>(args)
            }
            "u32" => {
                run_performance_test_generic::<PackedSparseDataset<DotVByteU32FixedU8Encoder>>(args)
            }
            _ => {
                eprintln!("Error: dotvbyte requires component-type 'u16' or 'u32'.");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.value_type() == "dotvbyte-scalar" {
        match args.component_type() {
            "u16" => {
                run_performance_test_generic::<PackedSparseDataset<DotVByteScalarU8Encoder>>(args)
            }
            "u32" => {
                run_performance_test_generic::<PackedSparseDataset<DotVByteU32ScalarU8Encoder>>(
                    args,
                )
            }
            _ => {
                eprintln!("Error: dotvbyte-scalar requires component-type 'u16' or 'u32'.");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.value_type() == "dotpacking8-fixedu8" {
        if args.component_type() != "u16" {
            eprintln!("Error: dotpacking8-fixedu8 requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<DotPacking8FixedU8Encoder>>(args);
        return;
    }

    if args.value_type() == "dotpacking8-scalar" {
        if args.component_type() != "u16" {
            eprintln!("Error: dotpacking8-scalar requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<DotPacking8ScalarU8Encoder>>(args);
        return;
    }

    if args.value_type() == "dotpacking8-sparse-scalar" {
        if args.component_type() != "u16" {
            eprintln!("Error: dotpacking8-sparse-scalar requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<DotPackingSparseScalarU8Encoder>>(args);
        return;
    }

    if args.value_type() == "dotpacking8-scalar4bit" {
        if args.component_type() != "u16" {
            eprintln!("Error: dotpacking8-scalar4bit requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<DotPacking8Scalar4BitEncoder>>(args);
        return;
    }

    if args.value_type() == "dotpacking8-centroid4bit" {
        if args.component_type() != "u16" {
            eprintln!("Error: dotpacking8-centroid4bit requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<DotPacking8Centroid4BitEncoder>>(args);
        return;
    }

    if args.value_type() == "dotvbyte-optimistic" {
        if args.component_type() != "u32" {
            eprintln!("Error: dotvbyte-optimistic requires component-type 'u32'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<OptimisticDotVByteFixedU8Encoder>>(args);
        return;
    }

    if args.value_type() == "dotvbyte-optimistic-scalar" {
        if args.component_type() != "u32" {
            eprintln!("Error: dotvbyte-optimistic-scalar requires component-type 'u32'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<OptimisticDotVByteScalarU8Encoder>>(
            args,
        );
        return;
    }

    if args.value_type() == "dotpacking8-optimistic-scalar" {
        if args.component_type() != "u32" {
            eprintln!("Error: dotpacking8-optimistic-scalar requires component-type 'u32'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<
            PackedSparseDataset<OptimisticDotPacking8U32ScalarU8Encoder>,
        >(args);
        return;
    }

    if args.value_type() == "dotpacking8-f16" {
        if args.component_type() != "u16" {
            eprintln!("Error: dotpacking8-f16 requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<DotPacking8f16Encoder>>(args);
        return;
    }

    if args.value_type() == "dotpackingsparse-f16" {
        if args.component_type() != "u16" {
            eprintln!("Error: dotpackingsparse-f16 requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<DotPackingSparsef16Encoder>>(args);
        return;
    }

    if args.value_type() == "packed-centroid" {
        if args.component_type() != "u16" {
            eprintln!("Error: packed-centroid requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<PackedCentroidSparseQuantizer>>(args);
        return;
    }

    if args.value_type() == "packed-variable-uniform" {
        if args.component_type() != "u16" {
            eprintln!("Error: packed-variable-uniform requires component-type 'u16'.");
            std::process::exit(1);
        }
        run_performance_test_generic::<PackedSparseDataset<PackedVariableBitUniformSparseQuantizer>>(
            args,
        );
        return;
    }

    macro_rules! run_performance_test_macro {
        ($C:ty, $V:ty) => {
            println!(
                "Using {} component type with {} value type",
                stringify!($C),
                stringify!($V)
            );
            run_performance_test_generic::<
                vectorium::PlainSparseDataset<$C, $V, vectorium::DotProduct>,
            >(args);
        };
    }

    match_component_value!(
        args.component_type(),
        args.value_type(),
        run_performance_test_macro
    );
}
