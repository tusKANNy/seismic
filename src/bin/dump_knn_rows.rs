//! Dumps the neighbour lists of a serialized kNN graph (`.knn.seismic`) as a TSV with rows
//! `doc_id \t rank \t neighbor_id`. The graph stores no scores, so there is no score column.
//!
//! Restricting the dump to a sampled subset of doc ids is what makes graph accuracy measurable:
//! the exact doc-to-doc groundtruth only exists for the sample, and dumping all 8.8M rows would
//! be gigabytes of TSV for nothing.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use clap::Parser;

use seismic::inverted_index::Knn;
use vectorium::IndexSerializer;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the `.knn.seismic` file.
    #[clap(short, long, value_parser)]
    knn_file: String,

    /// Optional text file with one doc id per line; dumps all docs if absent.
    #[clap(short, long, value_parser)]
    doc_ids: Option<String>,

    /// Output TSV path.
    #[clap(short, long, value_parser)]
    output_path: String,
}

pub fn main() {
    let args = Args::parse();

    let knn: Knn = Knn::load_index(&args.knn_file)
        .unwrap_or_else(|err| panic!("Failed to load knn graph from {}: {err:?}", args.knn_file));
    println!(
        "Loaded knn graph: {} vectors, {} neighbors each",
        knn.n_vecs(),
        knn.dim()
    );

    let doc_ids: Vec<usize> = match &args.doc_ids {
        Some(path) => BufReader::new(File::open(path).unwrap())
            .lines()
            .map(|line| line.unwrap().trim().parse().unwrap())
            .collect(),
        None => (0..knn.n_vecs()).collect(),
    };

    let mut output = BufWriter::new(File::create(&args.output_path).unwrap());
    for &doc_id in &doc_ids {
        for (rank, neighbour) in knn.neighbours_of(doc_id).into_iter().enumerate() {
            writeln!(&mut output, "{doc_id}\t{}\t{neighbour}", rank + 1).unwrap();
        }
    }
    println!(
        "Wrote {} rows to {}",
        doc_ids.len() * knn.dim(),
        args.output_path
    );
}
