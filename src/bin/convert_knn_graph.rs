//! Re-encodes a precomputed kNN graph from a numpy matrix into the current `.knn.seismic` format.
//!
//! Seismic 0.3.1 stored `Knn` as `{ n_vecs, d, neighbours: qwt::BitVector, nbits }`; 0.5.2 stores
//! `{ n_vecs, dim, neighbours: toolkit::BitField }`. The two serialise differently, so a graph
//! written by 0.3.1 loads into 0.5.2 without error — `n_vecs` and `dim` are both `usize` in the
//! same position and decode correctly — and then yields garbage neighbour ids, which the search
//! path reads with `get_unchecked` and segfaults on.
//!
//! Recomputing the graph with `Knn::new` would avoid the format problem but silently change the
//! experiment: it builds the graph with Seismic's own approximate search rather than whatever
//! produced the original. Re-encoding the matrix the original graph came from keeps the input
//! identical.
//!
//! Usage:
//!   convert_knn_graph --input <graph.npy> --output <prefix>   # writes <prefix>.knn.seismic

use std::fs::File;
use std::io::{BufReader, Read};

use clap::Parser;
use seismic::inverted_index::Knn;

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Numpy matrix of neighbour ids, shape (n_vecs, dim), C order, dtype uint32/int32/uint64.
    #[clap(short, long, value_parser)]
    input: String,

    /// Output prefix; the file is written as `<output>.knn.seismic`.
    #[clap(short, long, value_parser)]
    output: String,

    /// Keep only the first `dim` neighbours of each vector. Defaults to all of them.
    #[clap(long, value_parser)]
    dim: Option<usize>,
}

/// Element type of the numpy array, limited to what a neighbour id can be stored as.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Dtype {
    U32,
    I32,
    U64,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Dtype::U32 | Dtype::I32 => 4,
            Dtype::U64 => 8,
        }
    }
}

struct NpyHeader {
    dtype: Dtype,
    rows: usize,
    cols: usize,
}

/// Parses a .npy header, which is a magic string followed by an ASCII dict literal.
///
/// Only the cases a neighbour matrix can be in are accepted: C order, two dimensions, and an
/// integer element type wide enough to hold a document id. Anything else is an error rather
/// than a silent misread, because every failure mode here produces plausible-looking garbage.
fn read_npy_header(reader: &mut impl Read) -> NpyHeader {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).expect("cannot read npy magic");
    assert_eq!(&magic[..6], b"\x93NUMPY", "not a .npy file");

    let header_len = match magic[6] {
        1 => {
            let mut len = [0u8; 2];
            reader.read_exact(&mut len).expect("cannot read header length");
            u16::from_le_bytes(len) as usize
        }
        2 | 3 => {
            let mut len = [0u8; 4];
            reader.read_exact(&mut len).expect("cannot read header length");
            u32::from_le_bytes(len) as usize
        }
        version => panic!("unsupported npy version {version}"),
    };

    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header).expect("cannot read npy header");
    let header = String::from_utf8(header).expect("npy header is not valid UTF-8");

    let descr = between(&header, "'descr':", ",").replace(['\'', '"', ' '], "");
    let dtype = match descr.as_str() {
        "<u8" | "|u8" => Dtype::U64,
        "<u4" | "|u4" => Dtype::U32,
        "<i4" | "|i4" => Dtype::I32,
        other => panic!("unsupported dtype {other}: neighbour ids must be uint32/int32/uint64"),
    };

    assert!(
        between(&header, "'fortran_order':", ",").contains("False"),
        "the matrix must be in C order"
    );

    let shape = between(&header, "'shape':", ")");
    let dims: Vec<usize> = shape
        .trim_start_matches([' ', '('])
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("cannot parse shape"))
        .collect();
    assert_eq!(dims.len(), 2, "expected a 2-dimensional (n_vecs, dim) matrix");

    NpyHeader {
        dtype,
        rows: dims[0],
        cols: dims[1],
    }
}

fn between<'a>(haystack: &'a str, after: &str, before: &str) -> &'a str {
    let start = haystack.find(after).unwrap_or_else(|| panic!("missing {after} in npy header"))
        + after.len();
    let rest = &haystack[start..];
    let end = rest.find(before).unwrap_or(rest.len());
    &rest[..end]
}

fn main() {
    let args = Args::parse();

    let file = File::open(&args.input).expect("cannot open the input matrix");
    let mut reader = BufReader::with_capacity(1 << 22, file);
    let header = read_npy_header(&mut reader);

    let dim = args.dim.unwrap_or(header.cols);
    assert!(
        dim <= header.cols,
        "asked for {dim} neighbours per vector but the matrix only has {}",
        header.cols
    );

    println!(
        "Reading {} : {} vectors x {} neighbours ({:?}), keeping {dim} per vector",
        args.input, header.rows, header.cols, header.dtype
    );

    // Read row by row so that only the kept prefix of each row is retained: at 8.8M x 50 the
    // whole matrix is several GiB and the discarded columns need never be materialised.
    let element = header.dtype.size();
    let mut row = vec![0u8; header.cols * element];
    let mut neighbours = Vec::with_capacity(header.rows * dim);

    for id in 0..header.rows {
        reader
            .read_exact(&mut row)
            .unwrap_or_else(|err| panic!("truncated matrix at row {id}: {err}"));

        for i in 0..dim {
            let bytes = &row[i * element..(i + 1) * element];
            let value = match header.dtype {
                Dtype::U64 => u64::from_le_bytes(bytes.try_into().unwrap()),
                Dtype::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as u64,
                Dtype::I32 => {
                    let signed = i32::from_le_bytes(bytes.try_into().unwrap());
                    assert!(signed >= 0, "negative neighbour id {signed} at row {id}");
                    signed as u64
                }
            };
            assert!(
                (value as usize) < header.rows,
                "neighbour id {value} at row {id} is out of range for {} vectors",
                header.rows
            );
            neighbours.push(value);
        }
    }

    let knn = Knn::from_neighbours(header.rows, dim, neighbours);
    knn.serialize(&args.output).expect("cannot write the graph");
    println!("Done.");
}
