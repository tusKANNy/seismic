/// Full-scan brute-force search on the raw dataset to check DotVByteScalarU8 encoding quality.
///
/// Loads `documents.bin` directly, converts it to the target encoding, then does an exhaustive
/// scan over all documents for each query. No inverted index is involved.
///
/// This isolates whether the recall issue in seismic comes from:
/// - The encoding itself being too lossy  → full scan also shows low recall vs ground truth
/// - The block-skipping logic             → full scan shows ~100% but regular search does not
///
/// Output format is identical to `perf_inverted_index` so `scripts/recall.py` works unchanged.
use std::cmp;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use clap::Parser;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use vectorium::distances::DotProduct;
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::dotvbyte_scalaru8::DotVByteScalarU8Encoder;
use vectorium::{Dataset, Distance, PackedSparseDataset, PlainSparseDataset, read_seismic_format};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the dataset file (documents.bin, Seismic binary format).
    #[clap(short, long, value_parser)]
    input_file: String,

    /// Path to the query file (queries.bin, Seismic binary format).
    #[clap(short, long, value_parser)]
    query_file: String,

    /// Output file with ranked results (same tab-separated format as perf_inverted_index).
    #[clap(short, long, value_parser)]
    output_path: String,

    /// Number of queries to evaluate (capped at the number of available queries).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 6980)]
    n_queries: usize,

    /// Number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// Encoding to apply to the dataset before searching.
    /// `dotvbyte-scalar` = DotVByteScalarU8Encoder, `dotvbyte` = DotVByteFixedU8Encoder.
    #[clap(short, long, value_parser)]
    #[arg(default_value = "dotvbyte-scalar")]
    value_type: String,
}

fn pb_style() -> indicatif::ProgressStyle {
    ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})",
        )
        .unwrap()
        .progress_chars("=>-")
}

fn write_results(results: &[Vec<(f32, u64)>], output_path: &str) {
    let mut f = File::create(output_path).unwrap();
    for (query_id, top_k) in results.iter().enumerate() {
        for (rank, &(score, doc_id)) in top_k.iter().enumerate() {
            writeln!(f, "{query_id}\t{doc_id}\t{}\t{score}", rank + 1).unwrap();
        }
    }
    println!("Results written to {output_path}");
}

pub fn main() {
    let args = Args::parse();

    println!("Loading dataset...");
    let dataset_f32: PlainSparseDataset<u16, f32, DotProduct> =
        read_seismic_format(&args.input_file).expect("failed to read dataset");

    println!("Loading queries...");
    let queries: PlainSparseDataset<u16, f32, DotProduct> =
        read_seismic_format(&args.query_file).expect("failed to read queries");

    let n_queries = cmp::min(args.n_queries, queries.len());
    let k = args.k;

    println!("Dataset : {} docs", dataset_f32.len());
    println!("Queries : {} (using {})", queries.len(), n_queries);
    println!("Encoding: {}", args.value_type);

    match args.value_type.as_str() {
        "dotvbyte-scalar" => {
            println!("Converting dataset...");
            let dataset: PackedSparseDataset<DotVByteScalarU8Encoder> = dataset_f32.into();
            println!("Full-scan search (k={k})...");
            let start = Instant::now();
            let results: Vec<Vec<(f32, u64)>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(pb_style())
                .map(|qi| {
                    dataset
                        .search(queries.get(qi as u64), k)
                        .into_iter()
                        .map(|s| (s.distance.distance(), s.vector))
                        .collect()
                })
                .collect();
            println!(
                "Avg query time: {} µs",
                start.elapsed().as_micros() / n_queries as u128
            );
            write_results(&results, &args.output_path);
        }
        "dotvbyte" => {
            println!("Converting dataset...");
            let dataset: PackedSparseDataset<DotVByteFixedU8Encoder> = dataset_f32.into();
            println!("Full-scan search (k={k})...");
            let start = Instant::now();
            let results: Vec<Vec<(f32, u64)>> = (0..n_queries)
                .into_par_iter()
                .progress_count(n_queries as u64)
                .with_style(pb_style())
                .map(|qi| {
                    dataset
                        .search(queries.get(qi as u64), k)
                        .into_iter()
                        .map(|s| (s.distance.distance(), s.vector))
                        .collect()
                })
                .collect();
            println!(
                "Avg query time: {} µs",
                start.elapsed().as_micros() / n_queries as u128
            );
            write_results(&results, &args.output_path);
        }
        other => {
            eprintln!("Error: value-type must be 'dotvbyte-scalar' or 'dotvbyte'. Got: '{other}'");
            std::process::exit(1);
        }
    }
}
