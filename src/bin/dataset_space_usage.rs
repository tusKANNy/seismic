use clap::Parser;
use half::f16;
use std::fs::File;
use std::io::Write;

use vectorium::dataset::ConvertFrom;
use vectorium::{Dataset, DotProduct, ScalarSparseDataset, SpaceUsage, read_seismic_format};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Report the plain (uncompressed) forward index space usage of a Seismic binary dataset."
)]
pub struct Args {
    /// Path to the dataset file (e.g. `documents.bin`).
    #[clap(short, long)]
    input_file: String,

    /// Output TSV file to write the space usage to.
    /// If omitted, only prints to stdout.
    #[clap(short, long)]
    output_file: Option<String>,
}

fn main() {
    let args = Args::parse();

    // Try loading as u16 components first (most common case).
    // If that fails (e.g. vocabulary > 65K), fall back to u32.
    // Values are converted from f32 (on-disk) to f16 (in-memory) to match the
    // plain ScalarSparseDataset<C, f32, f16> baseline.
    let (component_type, n_vectors, n_dimensions, avg_nnz, space_bytes) =
        match read_seismic_format::<u16, f32, DotProduct>(&args.input_file) {
            Ok(dataset_f32) => {
                let n = dataset_f32.len();
                let dim = dataset_f32.input_dim();
                let nnz = dataset_f32.nnz();
                let dataset =
                    ScalarSparseDataset::<u16, f32, f16, DotProduct>::convert_from(dataset_f32);
                (
                    "u16",
                    n,
                    dim,
                    nnz as f64 / n as f64,
                    SpaceUsage::space_usage_bytes(&dataset),
                )
            }
            Err(e_u16) => {
                eprintln!("Failed to load as u16 ({}), trying u32...", e_u16);
                match read_seismic_format::<u32, f32, DotProduct>(&args.input_file) {
                    Ok(dataset_f32) => {
                        let n = dataset_f32.len();
                        let dim = dataset_f32.input_dim();
                        let nnz = dataset_f32.nnz();
                        let dataset =
                            ScalarSparseDataset::<u32, f32, f16, DotProduct>::convert_from(
                                dataset_f32,
                            );
                        (
                            "u32",
                            n,
                            dim,
                            nnz as f64 / n as f64,
                            SpaceUsage::space_usage_bytes(&dataset),
                        )
                    }
                    Err(e_u32) => {
                        eprintln!("Failed to load as u16: {}", e_u16);
                        eprintln!("Failed to load as u32: {}", e_u32);
                        std::process::exit(1);
                    }
                }
            }
        };

    println!("Dataset: {}", args.input_file);
    println!("Component type: {}", component_type);
    println!("Number of vectors: {}", n_vectors);
    println!("Number of dimensions: {}", n_dimensions);
    println!("Avg non-zero components: {:.2}", avg_nnz);
    println!("Forward index space usage: {} Bytes", space_bytes);
    println!(
        "Forward index space usage: {:.2} GB",
        space_bytes as f64 / 1024.0_f64.powi(3)
    );

    if let Some(output_path) = &args.output_file {
        let write_header = !std::path::Path::new(output_path).exists();
        let mut file = File::options()
            .create(true)
            .append(true)
            .open(output_path)
            .unwrap_or_else(|e| {
                eprintln!("Failed to open output file {}: {}", output_path, e);
                std::process::exit(1);
            });

        if write_header {
            writeln!(
                file,
                "dataset\tcomponent_type\tn_vectors\tn_dimensions\tavg_nnz\tspace_usage_bytes"
            )
            .unwrap();
        }

        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{:.2}\t{}",
            args.input_file, component_type, n_vectors, n_dimensions, avg_nnz, space_bytes
        )
        .unwrap();

        println!("Written to {}", output_path);
    }
}
