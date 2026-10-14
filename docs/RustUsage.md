## Using the Rust Code
This guide explains how to use Seismic's Rust code independently ([standalone](#itself)) or integrate it into your own Rust project ([via Cargo](#notitsef)).


### <a name="itself"> Using Seismic Binary Executable</a>
First, we clone the Seismic Git repository:

```bash
git clone git@github.com:TusKANNy/seismic.git
cd seismic
```

Then, we have to compile the project. After executing the following command, the binary executable will be found in `./target/release`. 

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

#### Building the Index
Let's now build an index on the Splade embeddings for the MS MARCO v1 passage. To download the encoded vectors, please refer to [Setting up for the Experiment](RunExperiments.md#bin_data).

Seismic has several parameters that control the space/time trade-offs when building the index:

- `-i, --input-file` path to the source collection file (`documents.bin`).
- `-o, --output-file` output index base path; the binary appends `.index.seismic`.
- `-n, --n-postings` defines the posting list size, representing the average number of postings stored per list (default: 6000).
- `--summarization-strategy` selects how each block is summarized for pruning at query time. Options: `energy-preserving` (default), `medoid`, `fixed-size`.
  - `energy-preserving`: the summary of a block keeps the largest components up to a fraction `--summary-energy` of the block energy.
  - `medoid`: no summary is built; the document of the block closest to the block centroid (its medoid) is stored apart from the rest of the block and scored against the query in place of a summary. Uses less space than a summary, with no quantization error (the medoid is a real document, so it is also returned with its exact score).
  - `fixed-size`: the summary keeps the `--n-components` largest components of the block.
- `-s, --summary-energy` controls the summary size for `energy-preserving`, preserving a fraction of the overall energy for each summary (default: 0.5).
- `--n-components` number of components of each summary for `fixed-size` (default: 0).
- `--centroid-fraction` determines the number of centroids per posting list, capped at a fraction of the posting list length (default: 0.1).
- `--clustering-algorithm` selects the algorithm to cluster postings within each posting list. Options: `random-kmeans`, `random-kmeans-inverted-index`, `random-kmeans-inverted-index-approx`.
- `--pruning-strategy` selects the posting list pruning strategy. Options: `global-threshold`, `coi-threshold`, `fixed-size`.
- `-b, --block-size` block size used for fixed-size blocking (default: 10).
- `--kmeans-doc-cut` number of top components retained while clustering (default: 15). Used by `random-kmeans-inverted-index` and `random-kmeans-inverted-index-approx`.
- `--kmeans-pruning-factor` pruning factor used by the random k-means blocking (default: 0.005). Used by `random-kmeans-inverted-index` only.
- `--min-cluster-size` minimum cluster size allowed (default: 2).
- `-a, --alpha` parameter of the `coi-threshold` pruning strategy (default: 0.15): each posting list keeps only its `alpha * len` largest entries, further capped at `n-postings`. This is the Set α-MSS of the paper; in expectation it preserves a fraction `alpha` of the L1 mass of each vector.
- `-m, --max-fraction` maximum posting list length as a factor of `n_postings` (default: 1.5).
- `--knn` number of neighbors stored per vector for the k-NN graph (default: 0, i.e., disabled).
- `--knn-path` path to a precomputed nearest-neighbor file.
- `--component-type` component type: `u16` (default) or `u32` for large vocabularies.
- `-v, --value-type` value type: `f16` (default), `bf16`, `f32`, `fixedu8`, `fixedu16`, `scalaru8`, `scalaru16`, or one of the packed formats below (the list, with the component types each one supports, is in `src/value_types.rs`). `f16`, `bf16` and `fixedu16` convert the values before building; every other format builds the index with plain `f32` values and converts the forward index at the end, so every format accepts the same build parameters. `perf_inverted_index` must be run with the same `--component-type` and `--value-type` used to build the index.
  - `scalaru8`, `scalaru16`: plain sparse vectors with 8/16-bit codes and one quantization step trained per component. `u16` and `u32`.
  - `blocked-sparse`, `blocked-sparse-scalaru8`, `blocked-sparse-scalaru16`: vectors packed in cache-line-aligned blocks with `f16` values, or 8/16-bit codes with one step per component. `u16` only.
  - `dotpacking8-fixedu8`, `dotpacking8-scalaru8`, `dotpacking8-scalaru16`, `dotpacking8-f16`, `dotpacking8-scalar4bit`: the DotPacking8 component codec (components reordered by decreasing document frequency and packed in 8-bit gaps) with the values stored as the suffix says: 8-bit fixed point, 8/16-bit codes with one step per component, `f16`, or 4-bit codes with one step per component. `u16`; `dotpacking8-scalaru8` and `dotpacking8-scalaru16` also take `u32`, for which they use the bridged encoder that keeps the 8-bit gaps and inserts bridge entries for the rare larger gaps. `dotpacking8-scalar` is the former name of `dotpacking8-scalaru8` and is still accepted.
  - Removed with vectorium 0.10: `dotvbyte*` (use `dotpacking8-scalaru8`), `dotpacking8-optimistic-scalar` (use `dotpacking8-scalaru8` with `u32`), `dotpacking8-sparse-scalar`, `dotpacking8-centroid4bit`, `dotpackingsparse-f16`, `packed-centroid`, `packed-variable-uniform`. Indexes serialized with them, and DotPacking8 indexes built before vectorium 0.10, must be rebuilt.

For **Splade on MS MARCO**, good choices are:
```
--n-postings 4000 --summary-energy 0.4 --centroid-fraction 0.1 --clustering-algorithm random-kmeans-inverted-index-approx
```

To create a Seismic index serialized in the file `documents.bin.4000_0.4_0.1.index.seismic`, run:

```bash
./target/release/build_inverted_index \
    -i ~/sparse_datasets/msmarco_v1_passage/cocondenser/data/documents.bin \
    -o ~/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/documents.bin.4000_0.4_0.1 \
    --centroid-fraction 0.1 \
    --summary-energy 0.4 \
    --n-postings 4000 \
    --clustering-algorithm random-kmeans-inverted-index-approx
```

#### Executing Queries
To query the index we need to use the `perf_inverted_index` executable. Its parameters are:

- `-i, --index-file` path to the serialized index file.
- `-q, --query-file` query file containing ground-truth vectors.
- `-o, --output-path` output file with the ranked results.
- `--query-cut` limits the search algorithm to the top `query_cut` components of the query (default: 10).
- `--heap-factor` skips blocks whose estimated dot product exceeds `heap_factor` times the smallest dot product in the top-k results (default: 0.7).
- `-k` number of top-k results to retrieve (default: 10).
- `--n-knn` number of neighbors of top results to rescore via the k-NN graph (default: 0).
- `-f, --first-sorted` whether to sort the first posting list by estimated dot products before search (default: false).
- `--n-queries` number of queries to evaluate (default: 10000).
- `-r, --n-runs` number of runs to average (default: 1).
- `--query-energy` optional query energy filter.
- `--component-type` component type: `u16` (default) or `u32`.
- `-v, --value-type` value type: `f16` (default), `bf16`, `f32`, `fixedu8`, or `fixedu16`.

Example command:

```bash
./target/release/perf_inverted_index \
    -i ~/sparse_datasets/msmarco_v1_passage/cocondenser/indexes/documents.bin.4000_0.4_0.1.index.seismic \
    -q ~/sparse_datasets/msmarco_v1_passage/cocondenser/data/queries.bin \
    -o results.tsv \
    --query-cut 5 \
    --heap-factor 0.7
```

The results are written to `results.tsv`. Each query produces `k` lines in the following format:

```text
query_id\tdocument_id\tresult_rank\tdot_product
```

Where:
- `query_id`: A progressive identifier for the query.
- `document_id`: The document ID from the indexed dataset.
- `result_rank`: The ranking of the result by dot product.
- `dot_product`: The similarity score.

---

## <a name="notitsef">Use Seismic in Your Rust Code</a>

To incorporate the Seismic library into your Rust project, navigate to your project directory and run:

```bash
cargo add seismic
```

### Creating a Toy Dataset
Let's create a toy dataset with vectors using `f32` values. We'll then convert it to use half-precision floating points ([`half::f16`](https://docs.rs/half/latest/half/)) and check the dataset's properties.

```rust
use seismic::SparseDataset;
use half::f16;

let data = vec![
    (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    (vec![1, 3],       vec![4.0, 5.0]),
    (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
];

let dataset: SparseDataset<f16> = data.into_iter().collect::<SparseDataset<f32>>().into();

assert_eq!(dataset.len(), 3);  // Number of vectors  
assert_eq!(dataset.dim(), 5);  // Number of components  
assert_eq!(dataset.nnz(), 9);  // Number of non-zero components  
```

To read a dataset in Seismic's internal binary format and quantize values to `f16`:

```rust
let dataset = SparseDataset::<f32>::read_bin_file(&input_filename).unwrap().quantize_f16();
```

### Building and Querying an Index
Let's build an index using our toy dataset and search for a query.

```rust
use seismic::inverted_index::{Configuration, InvertedIndex};
use seismic::SparseDataset;
use half::f16;

let data = vec![
    (vec![0, 2, 4],    vec![1.0, 2.0, 3.0]),
    (vec![1, 3],       vec![4.0, 5.0]),
    (vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0])
];

let dataset: SparseDataset<f16> = data.into_iter().collect::<SparseDataset<f32>>().into();

let inverted_index = InvertedIndex::build(dataset, Configuration::default());

let result = inverted_index.search(&vec![0, 1], &vec![1.0, 2.0], 1, 5, 0.7);

assert_eq!(result[0].0, 8.0);
assert_eq!(result[0].1, 1);
```

### Configuration Parameters
Key parameters to experiment with:
- `n_postings` (`PruningStrategy::GlobalThreshold`) defines the posting list size.
- `summary_energy` (`SummarizationStrategy::EnergyPreserving`) controls summary size; `SummarizationStrategy::Medoid` replaces the summaries with the medoid of each block, and `SummarizationStrategy::FixedSize` keeps `n_components` per summary.
- `centroid_fraction` (`BlockingStrategy::RandomKmeans`) sets centroids per posting list.

For serialization/deserialization examples, see:
- [`build_inverted_index.rs`](src/bin/build_inverted_index.rs)
- [`perf_inverted_index.rs`](src/bin/perf_inverted_index.rs)

The `search` method signature:

```rust
pub fn search(
    &self,
    query: SparseVectorView<'_, ComponentFor<S>, f32>,
    k: usize,
    query_cut: usize,
    heap_factor: f32,
    n_knn: usize,
    first_sorted: bool,
) -> Vec<ScoredVectorDotProduct>
```
