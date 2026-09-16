use std::cmp;
use std::fs::File;
use std::hash::Hash;
use std::io::Write;
use std::time::Instant;

use clap::Parser;
use num_traits::FromPrimitive;

use seismic::InvertedIndexBase;
use seismic::index_traits::{ComponentFor, EncoderFor};
use seismic::inverted_index::Knn;
use seismic::utils::energy_preserving_cut;
use vectorium::IndexSerializer;

use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::{
    ComponentType, Dataset, Distance, DotProduct, PackedSparseDataset, SpaceUsage,
    SparseVectorView, VectorEncoder, read_seismic_format,
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

    /// Serialized kNN graph (`.knn.seismic`) to attach to the loaded index, replacing any graph
    /// the index was built with. `--n-knn` then selects how many of its neighbours to rescore.
    ///
    /// Attaching at load time is what lets one index be measured against several graphs without
    /// a rebuild per (index, graph) pair.
    #[clap(long, value_parser)]
    knn_path: Option<String>,

    /// Whether to sort the first component by estimated dot products before search (see docs/RustUsage.md#using-the-rust-code).
    #[clap(short, long, action)]
    #[arg(default_value_t = false)]
    first_sorted: bool,

    /// Energy-preserving query pruning. When set, the number of query components to scan is
    /// chosen per query as the smallest prefix (by decreasing value) whose cumulative mass
    /// reaches this fraction of the query's total mass, overriding `query_cut`.
    #[clap(long, value_parser)]
    query_energy: Option<f32>,

    /// Component type (`u16` or `u32`); see docs/RustUsage.md#using-the-rust-code for guidance.
    #[clap(long, value_parser)]
    #[arg(default_value = "u16")]
    component_type: String,

    /// Value type (`f16`, `bf16`, `f32`, `fixedu8`, or `fixedu16`); see docs/RustUsage.md#using-the-rust-code for recommendations.
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
                "Error: component-type must be either 'u16' or 'u32', value-type must be 'f16', 'bf16', 'f32', 'fixedu16', or 'fixedu8'"
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

    let mut inverted_index: InvertedIndexBase<S> =
        InvertedIndexBase::load_index(index_path.unwrap().as_str())
            .unwrap_or_else(|err| panic!("Failed to load index: {err:?}"));

    if let Some(knn_path) = &args.knn_path {
        inverted_index.add_knn(Knn::new_from_serialized(knn_path, None));
    }

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

    match args.query_energy {
        Some(energy_fraction) => println!(
            "Query pruning: energy-preserving, fraction {energy_fraction} (query-cut ignored)"
        ),
        None => println!("Query pruning: fixed, top-{query_cut} components"),
    }

    let mut results = Vec::with_capacity(n_queries);
    let mut energy_buf = Vec::new();

    let time = Instant::now();
    for _ in 0..n_runs {
        results.clear();

        for (query_id, components_values) in queries.iter().take(n_queries).enumerate() {
            let query_view =
                SparseVectorView::new(components_values.components(), components_values.values());
            let query = query_view;

            let effective_query_cut = match args.query_energy {
                Some(energy_fraction) => energy_preserving_cut(
                    components_values.values(),
                    energy_fraction,
                    &mut energy_buf,
                ),
                None => query_cut,
            };

            let cur_results = inverted_index.search(
                query,
                args.k,
                effective_query_cut,
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

#[cfg(test)]
mod tests {
    use super::energy_preserving_cut;

    /// The reference implementation this port reproduces, transcribed from the version of
    /// `InvertedIndex::search` that produced the submitted numbers. It sorts (component, value)
    /// pairs and returns the prefix length, which is what the port computes from values alone.
    fn reference_cut(query_values: &[f32], energy_fraction: f32) -> usize {
        let total_energy = query_values.iter().sum::<f32>();
        let mut sorted_tuples: Vec<(usize, f32)> =
            query_values.iter().copied().enumerate().collect();
        sorted_tuples.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut accumulated_energy = 0.0;
        let mut break_index = sorted_tuples.len();
        for (i, (_comp, val)) in sorted_tuples.iter().enumerate() {
            if (accumulated_energy + val) / total_energy >= energy_fraction {
                break_index = i + 1;
                break;
            }
            accumulated_energy += val;
        }
        break_index
    }

    fn cut(query_values: &[f32], energy_fraction: f32) -> usize {
        energy_preserving_cut(query_values, energy_fraction, &mut Vec::new())
    }

    #[test]
    fn takes_smallest_prefix_reaching_the_requested_mass() {
        // Uniform values: half the mass needs half the components.
        assert_eq!(cut(&[1.0, 1.0, 1.0, 1.0], 0.5), 2);
        // The full mass needs every component.
        assert_eq!(cut(&[1.0, 1.0, 1.0, 1.0], 1.0), 4);
        // A single dominant component already covers the requested fraction.
        assert_eq!(cut(&[0.1, 0.7, 0.2], 0.6), 1);
        assert_eq!(cut(&[0.1, 0.7, 0.2], 0.85), 2);
    }

    #[test]
    fn considers_components_in_decreasing_order_of_value() {
        // Input order must not matter: only the sorted mass profile does.
        assert_eq!(cut(&[0.1, 0.7, 0.2], 0.6), cut(&[0.7, 0.2, 0.1], 0.6));
        assert_eq!(cut(&[0.2, 0.1, 0.7], 0.6), cut(&[0.7, 0.2, 0.1], 0.6));
    }

    #[test]
    fn degenerate_queries_fall_back_to_the_whole_query() {
        assert_eq!(cut(&[], 0.5), 0);
        // A zero total makes the fraction undefined; the reference scans everything.
        assert_eq!(cut(&[0.0, 0.0, 0.0], 0.5), 3);
    }

    #[test]
    fn matches_the_reference_implementation() {
        let queries: [&[f32]; 6] = [
            &[1.0, 1.0, 1.0, 1.0],
            &[0.1, 0.7, 0.2],
            &[2.5, 0.01, 1.25, 0.4, 0.9, 0.05],
            &[0.3],
            &[],
            &[0.0, 0.0, 0.0],
        ];

        for query in queries {
            for step in 1..=10 {
                let energy_fraction = step as f32 / 10.0;
                assert_eq!(
                    cut(query, energy_fraction),
                    reference_cut(query, energy_fraction),
                    "mismatch for query {query:?} at energy {energy_fraction}"
                );
            }
        }
    }

    #[test]
    fn scratch_buffer_is_reusable_across_queries() {
        let mut buf = Vec::new();
        assert_eq!(energy_preserving_cut(&[0.1, 0.7, 0.2], 0.6, &mut buf), 1);
        // A shorter query after a longer one must not see stale values.
        assert_eq!(energy_preserving_cut(&[1.0, 1.0], 1.0, &mut buf), 2);
        assert_eq!(energy_preserving_cut(&[0.1, 0.7, 0.2], 0.6, &mut buf), 1);
    }
}

pub fn main() {
    let args = Args::parse();

    if args.value_type() == "dotvbyte" {
        if args.component_type() != "u16" {
            eprintln!("Error: dotvbyte is only supported with component-type 'u16'.");
            std::process::exit(1);
        }

        run_performance_test_generic::<PackedSparseDataset<DotVByteFixedU8Encoder>>(args);
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
