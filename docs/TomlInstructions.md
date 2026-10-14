# TOML Configuration File Documentation

This document describes the structure and parameters of TOML configuration files used to run experiments with the Seismic search engine. These configuration files define all aspects of an experiment, from dataset paths to indexing parameters and query configurations.

## Running Experiments

### Single Experiments
To run a single experiment using a TOML configuration file, use the Python script:

```bash
python scripts/run_experiments.py --exp path/to/your/config.toml
```

This script will:
1. Build the search index using parameters from `[indexing_parameters]`
2. Execute queries with different configurations from `[query]` section
3. Measure query time, recall, ranking metrics, and memory usage
4. Generate a `report.tsv` file with all results

For complete setup instructions and detailed examples, see [RunExperiments.md](RunExperiments.md).

### Grid Search Experiments
To perform parameter sweeps across multiple values, use the grid search script:

```bash
python scripts/run_grid_search.py --exp path/to/grid_search_config.toml
```

Grid search configurations use array values in `[indexing_parameters]` and `[querying_parameters]` sections to test all combinations of specified parameters. This generates multiple experiments automatically. See [Grid Search Configuration](#grid-search-configuration) for a description of these parameters.

**Example**: A configuration with `n-postings = [1000, 2000]` and `query-cut = [3, 5]` will run 4 experiments (2×2 combinations).

---

## File Structure Overview

Each TOML configuration file consists of the following main sections:

1. **Root-level metadata** - Experiment identification and optional build commands
2. **`[settings]`** - General experiment settings
3. **`[folder]`** - Directory paths for data, indexes, and results
4. **`[filename]`** - Specific file names within the directories
5. **`[indexing_parameters]`** - Parameters for building the search index
6. **`[query]`** - Query configurations for different recall targets
7. **`[querying_parameters]`** - Alternative query parameter format for grid searches

## Root-Level Metadata

These parameters identify and describe the experiment:

```toml
name =          "experiment_identifier"
title =         "Human-readable experiment title"
description =   "Detailed description of the experiment purpose"
dataset =       "Dataset description (e.g., 'Splade cocondenser on MS-MARCO')"
```

**Parameters:**
- `name`: Unique identifier for the experiment (used in output file names)
- `title`: Short, descriptive title for the experiment
- `description`: Detailed explanation of what the experiment tests
- `dataset`: Description of the dataset and embedding model used
- `compile-command`: Optional custom compilation command (commented by default)
- `build-command`: Optional custom index building command (commented by default)
- `query-command`: Optional custom query execution command (commented by default)
- `space-usage-command`: Optional custom path of the `dataset_space_usage` binary, used by the grid search only (default: `./target/release/dataset_space_usage`). Set it when `compile-command` builds the binaries elsewhere (e.g. `./target/debug/dataset_space_usage`)

## Settings Section

General configuration for the experiment execution:

```toml
[settings]
k =             10          # Number of top results to retrieve
n-runs =        1           # Number of experimental runs (affects query time measurement)
build =         true        # Whether to build the index (false if index already exists)
delete =        false       # Whether to remove index files after use to save disk space
metric =        "RR@10"     # Evaluation metric to compute
component-type = "u16"      # Component type for sparse vector indices
value-type =     "f16"      # Value type for sparse vector values (options: "f16", "bf16", "f32", "fixedu8", "fixedu16", or a compressed format, see below)
# value-type =   "fixedu8"  # Example: use 8-bit fixed-point quantization
# value-type =   "fixedu16" # Example: use 16-bit fixed-point quantization
# NUMA =        "numactl --physcpubind='0-15' --localalloc"  # NUMA configuration
```

**Parameters:**
- `k`: Number of top search results to retrieve per query (typically 10 or 100)
- `n-runs`: Number of times to repeat the experiment 
- `build`: Whether to build a new index (`true`) or use existing index (`false`)
- `metric`: Evaluation metric (common values: `"RR@10"`, `"MAP"`, `"NDCG@10"`)
- `component-type`: Data type for component indices in sparse vectors
  - `"u16"`: 16-bit unsigned integers (supports up to 65,535 unique components)
  - `"u32"`: 32-bit unsigned integers (supports up to 4+ billion unique components)

  - **Recommendation**: Use `"u16"` for most datasets unless you have more than 65K vocabulary terms
- `value-type`: Data type for storing vector values
  - `"f16"`: IEEE 754 half-precision (16-bit) floating point - **recommended for best performance**
  - `"bf16"`: Google Brain's bfloat16 format (16-bit) - alternative half-precision format
  - `"f32"`: IEEE 754 single-precision (32-bit) floating point - highest precision but slower
  - `"fixedu8"`: 8-bit fixed-point quantization (Q0.8) - very compact, lowest memory usage, may reduce accuracy
  - `"fixedu16"`: 16-bit fixed-point quantization (Q0.16) - compact, higher accuracy than fixedu8, lower memory than floats
  - `"scalaru8"`, `"scalaru16"`: 8/16-bit codes with one quantization step trained per component - `"u16"` or `"u32"`
  - Packed formats, each available only for the `component-type` listed (see [RustUsage.md](RustUsage.md#building-the-index) for what each one stores):
    - `"dotpacking8-scalaru8"`, `"dotpacking8-scalaru16"`: `"u16"` or `"u32"`
    - `"dotpacking8-fixedu8"`, `"dotpacking8-f16"`, `"dotpacking8-scalar4bit"`, `"blocked-sparse"`, `"blocked-sparse-scalaru8"`, `"blocked-sparse-scalaru16"`: `"u16"` only
  - `"dotvbyte"` and the other formats removed with vectorium 0.10 are rejected by `run_experiments.py` with the name of their replacement.

  - **Recommendation**: Use `"f16"` for optimal balance of speed, memory efficiency, and accuracy. Use `"fixedu8"` for maximum memory savings when some loss in accuracy is acceptable.
- `NUMA`: Optional NUMA (Non-Uniform Memory Access) configuration string for performance optimization

## Folder Section

Directory paths for all experiment data:

```toml
[folder]
data =          "~/sparse_datasets/msmarco_v1_passage/cocondenser/data"
index =         "~/sparse_datasets/msmarco_v1_passage/cocondenser/indexes"
qrels_path =    "~/sparse_datasets/msmarco_v1_passage/qrels.dev.small.tsv"
experiment =    "."         # Output directory for experiment results
```

**Parameters:**
- `data`: Directory containing the dataset files (documents and queries)
- `index`: Directory where search indexes are stored or will be created
- `qrels_path`: Path to the relevance judgments file (ground truth for evaluation)
- `experiment`: Directory where experiment outputs (logs, results) will be saved

## Filename Section

Specific file names within the data directories:

```toml
[filename]
dataset =       "documents.bin"      # Binary file containing document vectors
queries =       "queries.bin"        # Binary file containing query vectors
groundtruth =   "groundtruth.tsv"    # Tab-separated ground truth file
doc_ids =       "doc_ids.npy"        # NumPy array of document IDs
query_ids =     "queries_ids.npy"    # NumPy array of query IDs
index =         "experiment_index_name"  # Base name for index files
knn_path =      "knn_graph.npy"      # Optional: k-NN graph file (for KNN-enhanced indexes)
```

**Parameters:**
- `dataset`: Binary file containing sparse document representations
- `queries`: Binary file containing sparse query representations
- `groundtruth`: TSV file with query-document relevance pairs
- `doc_ids`: NumPy array mapping internal document IDs to external IDs
- `query_ids`: NumPy array mapping internal query IDs to external IDs
- `index`: Base name for generated index files (extensions will be added automatically)
- `knn_path`: Optional k-NN graph file for enhanced retrieval accuracy

## Indexing Parameters Section

Parameters controlling how the search index is built:

```toml
[indexing_parameters]
n-postings =            4000        # Target number of postings per posting list
centroid-fraction =     0.1         # Fraction of posting list length used for centroids
summary-energy =        0.4         # Fraction of energy preserved in summaries
# summarization-strategy = "medoid" # How blocks are summarized: "energy-preserving" (default) or "medoid"
knn =                   0           # Number of k-NN neighbors (0 = disabled)
clustering-algorithm =  "random-kmeans"  # Clustering algorithm choice
kmeans-doc-cut =        15          # Document cut parameter for kmeans variants
kmeans-pruning-factor = 0.005       # Pruning factor for inverted index kmeans
```

### Pruning Strategy Parameters
```toml
pruning-strategy =      "global-threshold"  # Pruning strategy choice
alpha =                 0.2                 # coi-threshold parameter: fraction of each posting list kept
# max-fraction =          1.5               # Global threshold parameter
```

**Core Parameters:**
- `n-postings`: Target average number of postings (document IDs) per posting list. Higher values = more memory, potentially better accuracy
- `centroid-fraction`: Controls number of cluster centroids as a fraction of posting list length (0.0-1.0)
- `summary-energy`: Fraction of total energy preserved in document summaries (0.0-1.0). Higher = better accuracy, more memory. Used by the `"energy-preserving"` summarization only; it can be omitted with the other strategies
- `summarization-strategy`: How each block of a posting list is summarized for pruning at query time. Optional, default `"energy-preserving"`
  - `"energy-preserving"`: summary made of the largest components of the block, up to `summary-energy`
  - `"medoid"`: no summary; the document of the block closest to its centroid (the medoid) is stored apart and scored in place of the summary. Less memory than a summary, no quantization error
  - The `fixed-size` summarization of `build_inverted_index` is not reachable from the scripts, since they do not pass `--n-components`
- `knn`: Number of k-nearest neighbors to store per document (0 = disabled, >0 enables KNN enhancement)

**Clustering Algorithm Options:**
- `"random-kmeans"`: Basic random k-means clustering
- `"random-kmeans-inverted-index"`: K-means with inverted index acceleration
- `"random-kmeans-inverted-index-approx"`: Approximate version of above (faster)

**Clustering-Specific Parameters:**
- `kmeans-doc-cut`: Number of top document components to consider during clustering (for `random-kmeans-inverted-index` and `random-kmeans-inverted-index-approx`)
- `min-cluster-size`: Minimum size of clusters during k-means clustering. Clusters smaller than this are removed and their documents are reassigned to the nearest valid cluster. This helps eliminate noise and improves clustering quality
- `kmeans-pruning-factor`: Pruning factor for `random-kmeans-inverted-index`

**Pruning Strategy Options:**
- `"global-threshold"`: Global threshold-based pruning (requires `max-fraction`)
- `"coi-threshold"`: per-list pruning, the Set α-MSS of the paper (requires `alpha`; `n-postings` acts as a cap)
- `"fixed-size"`: Fixed-size pruning

**Pruning Parameters:**
- `alpha`: `coi-threshold` parameter (0.0-1.0) - each posting list keeps only its `alpha * len` largest entries, capped at `n-postings`. In expectation this preserves a fraction `alpha` of the L1 mass of each vector
- `max-fraction`: Maximum posting list length as multiple of `n-postings` (e.g., 1.5 = 150% of n-postings)

## Query Section

Multiple query configurations that may achieve different recall targets:

```toml
[query]
    [query.recall_90]
    query-cut =         3
    heap-factor =       0.9
    knn =               0           # Optional: KNN neighbors for this query
    first-sorted =      true        # Optional: Whether to sort the inner product of the summaries 

    [query.recall_91]
    query-cut =         4
    heap-factor =       0.9
    # ... additional configurations
```

**Parameters:**
- `query-cut`: Maximum number of query components to process (1-∞). Lower = faster, potentially less accurate
- `heap-factor`: Pruning threshold multiplier (0.0-1.0). Lower = more aggressive pruning, faster queries
- `knn`: Optional number of KNN neighbors to use for this specific query configuration. This may boost recall *BUT* increases both space usage and building time significantly.
- `first-sorted`: Optional boolean indicating whether to sort the first posting list by block estimated inner product. This may speed up query time on some datasets.

**Naming Convention:**
Query sections can use any descriptive name following the pattern `[query.section_name]`. Common conventions include recall targets (e.g., `[query.recall_90]`, `[query.recall_95]`) or descriptive names (e.g., `[query.fast]`, `[query.accurate]`, `[query.balanced]`).

## Example Configurations

For pre-optimized configurations across several datasets and memory budgets, see [`experiments/best_configs/`](../experiments/best_configs/). These configurations demonstrate various parameter combinations and were obtained through extensive grid searches. See [BestResults.md](BestResults.md) for details.

---

## Grid Search Configuration

To run a full grid search over indexing and querying parameters, use the script:

```bash
python scripts/run_grid_search.py --exp path/to/grid_search_config.toml
```

This will:
- Read the grid specification from the TOML file
- Generate all valid combinations of indexing and querying parameters (via Cartesian product)
- Launch one experiment per combination

In the toml file we can spacify parameter sweeps for both indexing and querying parameters. 
Here **`experiments/grid_searches/`** is an example grid search configurations demonstrating parameter sweeps.

For parameter sweeps, use array values in `[indexing_parameters]` and `[querying_parameters]`:

```toml
[indexing_parameters]
n-postings =            [1000, 2000, 4000]    # Multiple values to test
centroid-fraction =     [0.1]                 # Single value (no sweep)
summary-energy =        [0.4, 0.5]            # Two values to test
clustering-algorithm =  ["random-kmeans"]     # String arrays supported
# summarization-strategy = ["medoid"]         # Summarization sweep (summary-energy is ignored by "medoid")

[querying_parameters]
query-cut =             [1, 2, 3, 4, 5]       # Query cut sweep
heap-factor =           [0.7, 0.8, 0.9, 1.0]  # Heap factor sweep
knn =                   [0]                    # KNN neighbors
first_sorted =          [true, false]         # Boolean options
```

This generates a Cartesian product of all parameter combinations.
The experiments will be saved in the folder specified in the TOML file. 

Once all combinations have been executed, the results can be analyzed using the Jupyter notebooks in `scripts/Notebooks/`:

### Collecting and Analyzing Grid Search Reports

After running a grid search with `scripts/run_grid_search.py`, each experiment produces multiple subdirectories (e.g., `building_combination_0_*/`) containing individual `report.tsv` files and `experiment_config.toml` files.

Two notebooks are provided to aggregate and analyze these results:

- **[`scripts/Notebooks/ExtractResutsGridSearch.ipynb`](../scripts/Notebooks/ExtractResutsGridSearch.ipynb)** - Aggregates all `report.tsv` files from a grid search run into a single DataFrame, enriches them with the indexing and query parameters from the corresponding TOML configs, and plots recall vs. query time curves for different memory budgets. Set the `base_dir` variable to your grid search output folder and `forward_index_size` to the size of your forward index in bytes.

- **[`scripts/Notebooks/DumpBestConfiguration.ipynb`](../scripts/Notebooks/DumpBestConfiguration.ipynb)** - Given an aggregated grid search DataFrame, extracts the fastest configuration for each recall level (90%–99%) within a given memory budget, and dumps the results as ready-to-use TOML files under `experiments/best_configs/`. This is the notebook used to generate the pre-optimized configurations shipped with the repository.
