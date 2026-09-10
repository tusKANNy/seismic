//! Builds a standalone kNN graph from an already-built serialized index.
//!
//! The graph is written to `<output-path>.knn.seismic` and can later be attached to any index
//! built on the same collection, either at build time (`build_inverted_index --knn-path`) or at
//! load time (`perf_inverted_index --knn-path`). Keeping it standalone is what makes a study of
//! "which index should build the graph" affordable: the graph is built once per construction
//! index, and every serving index reuses it without a rebuild.
//!
//! `--query-cut` / `--heap-factor` / `--first-sorted` expose the construction search parameters,
//! which `Knn::new` otherwise hardcodes to 10 / 0.7 / false. Those defaults were chosen for the
//! `global-threshold` index family; on another family a different setting can reach a
//! higher-accuracy graph, or the same accuracy for less work. Omitting all three reproduces
//! `Knn::new` exactly.
//!
//! `--query-energy` replaces the fixed cut with the energy-preserving query pruning of the COI
//! variant (the same rule `perf_inverted_index --query-energy` applies to queries): each document
//! is cut at the smallest prefix of its largest values holding that fraction of its mass. It is
//! the setting to use when the graph is meant to be compared with COI query pruning, whose only
//! query-side knob is that fraction; `--query-cut` is ignored when it is given.

use std::hash::Hash;
use std::time::Instant;

use clap::Parser;
use num_traits::FromPrimitive;

use seismic::InvertedIndexBase;
use seismic::index_traits::{ComponentFor, EncoderFor};
use seismic::inverted_index::Knn;
use vectorium::{ComponentType, IndexSerializer, SpaceUsage, SparseVectorView, VectorEncoder};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the serialized index file (`.index.seismic`).
    #[clap(short, long, value_parser)]
    index_file: String,

    /// Number of neighbors to store per vector.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 50)]
    nknn: usize,

    /// Output path; `.knn.seismic` is appended.
    #[clap(short, long, value_parser)]
    output_path: String,

    /// Query cut used to search each document during construction.
    #[clap(long, value_parser)]
    #[arg(default_value_t = Knn::KNN_QUERY_CUT)]
    query_cut: usize,

    /// Energy fraction of the energy-preserving query pruning applied to each document during
    /// construction (COI's alpha_q). Overrides `--query-cut`.
    #[clap(long, value_parser)]
    query_energy: Option<f32>,

    /// Heap factor used to search each document during construction.
    #[clap(long, value_parser)]
    #[arg(default_value_t = Knn::KNN_HEAP_FACTOR)]
    heap_factor: f32,

    /// Sort the first component by estimated dot products during construction.
    #[clap(long, action)]
    #[arg(default_value_t = false)]
    first_sorted: bool,

    /// Component type (`u16` or `u32`).
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type (`f16`, `bf16`, or `f32`).
    #[clap(short, long, value_parser)]
    #[arg(default_value = "f16")]
    value_type: String,
}

pub fn build_knn_generic<S>(args: Args)
where
    S: seismic::SeismicBuildDataset
        + seismic::SeismicSearchDataset
        + SpaceUsage
        + serde::Serialize
        + serde::de::DeserializeOwned,
    for<'a> <EncoderFor<S> as VectorEncoder>::QueryVector<'a>:
        From<SparseVectorView<'a, ComponentFor<S>, f32>>,
    ComponentFor<S>: ComponentType
        + FromPrimitive
        + SpaceUsage
        + Hash
        + serde::Serialize
        + serde::de::DeserializeOwned,
{
    let inverted_index: InvertedIndexBase<S> = InvertedIndexBase::load_index(&args.index_file)
        .unwrap_or_else(|err| panic!("Failed to load index: {err:?}"));

    println!("Number of documents: {}", inverted_index.len());
    match args.query_energy {
        Some(energy) => println!(
            "Construction search: query-energy {}, heap-factor {}, first-sorted {}",
            energy, args.heap_factor, args.first_sorted
        ),
        None => println!(
            "Construction search: query-cut {}, heap-factor {}, first-sorted {}",
            args.query_cut, args.heap_factor, args.first_sorted
        ),
    }

    let time = Instant::now();
    let knn = Knn::new_with_params(
        &inverted_index,
        args.nknn,
        args.query_cut,
        args.query_energy,
        args.heap_factor,
        args.first_sorted,
    );
    println!("Time to build knn graph: {} secs", time.elapsed().as_secs());

    knn.serialize(&args.output_path).unwrap();
}

pub fn main() {
    let args = Args::parse();

    macro_rules! build_knn_macro {
        ($C:ty, $V:ty) => {
            println!(
                "Using {} component type with {} value type",
                stringify!($C),
                stringify!($V)
            );
            build_knn_generic::<vectorium::PlainSparseDataset<$C, $V, vectorium::DotProduct>>(args);
        };
    }

    use half::{bf16, f16};
    match (args.component_type.as_str(), args.value_type.as_str()) {
        ("u16", "f16") => {
            build_knn_macro!(u16, f16);
        }
        ("u16", "bf16") => {
            build_knn_macro!(u16, bf16);
        }
        ("u16", "f32") => {
            build_knn_macro!(u16, f32);
        }
        ("u32", "f16") => {
            build_knn_macro!(u32, f16);
        }
        ("u32", "bf16") => {
            build_knn_macro!(u32, bf16);
        }
        ("u32", "f32") => {
            build_knn_macro!(u32, f32);
        }
        _ => {
            eprintln!(
                "Error: component-type must be 'u16' or 'u32', value-type must be 'f16', 'bf16', or 'f32'"
            );
            std::process::exit(1);
        }
    }
}
