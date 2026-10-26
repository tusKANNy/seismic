# Changes

## Unreleased

- vectorium 0.11.0 (`ab18e9e`) → 0.11.1 (tag `v0.11.1`, `406640c`). vectorium now takes the
  forward-index prefetch cap from the caller; Seismic passes 512 bytes, the value 0.11.0 hard-coded,
  so search behaviour is unchanged. Same vectorium as kANNolo 0.14+, so a Seismic index can be the
  first stage of kANNolo's `RerankIndex`. Indexes built with 0.6.0 load unchanged.

## 0.6.0

Seismic moves to vectorium 0.11.0 (`main`, pinned at `ab18e9e`), the release of its `compression`
branch, which drops the research-only encoders and adds DotPacking8, blocked and step-scalar sparse
encoders.

### Breaking changes

- **Removed value types.** The value type (`--value-type` of the binaries, `value-type` in the
  experiment TOMLs) selects how an index stores its values. vectorium 0.11 deleted some encoders,
  so these names are no longer accepted:

  | Removed value type | Use instead |
  |---|---|
  | `dotvbyte`, `dotvbyte-scalar`, `dotvbyte-optimistic`, `dotvbyte-optimistic-scalar` | `dotpacking8-scalaru8` |
  | `dotpacking8-optimistic-scalar` | `dotpacking8-scalaru8` with component type `u32` |
  | `dotpacking8-sparse-scalar`, `dotpacking8-centroid4bit`, `dotpackingsparse-f16`, `packed-centroid`, `packed-variable-uniform` | none (research-only encoders) |

  There is no silent fallback: the binaries and `scripts/run_experiments.py` stop with an error
  that names the replacement, e.g. `'dotvbyte': DotVByte was removed in vectorium 0.11; use
  'dotpacking8-scalaru8' (u16 or u32 components)`. `dotpacking8-scalaru8` stores 8-bit codes with
  one quantization step per component; against `dotvbyte` it is as fast or faster, uses 4–8% less
  memory and has slightly higher recall.
- **Python:** `SeismicIndexDotVByte` is renamed `SeismicIndexDotPacking8` (DotPacking8, 8-bit codes
  with one step per component). The old name is kept as a placeholder whose `build`,
  `build_from_dataset`, `load` and constructor raise a `RuntimeError` naming the replacement.
- **Indexes built with 0.5.x must be rebuilt.**
- **Rust API:** the generic bounds follow vectorium 0.11. Encoders must have `QueryParams = ()`
  and conversions `Config = ()`.

### Value types

Every value type of vectorium 0.11 can be used to build and query an index:

| Value type | Component types |
|---|---|
| `f32`, `f16`, `bf16`, `fixedu8`, `fixedu16` | `u16`, `u32` |
| `scalaru8`, `scalaru16` (new: one quantization step per component) | `u16`, `u32` |
| `blocked-sparse`, `blocked-sparse-scalaru8`, `blocked-sparse-scalaru16` (the last two new) | `u16` |
| `dotpacking8-scalaru8`, `dotpacking8-scalaru16` (the latter new) | `u16`, `u32` (new) |
| `dotpacking8-fixedu8`, `dotpacking8-f16`, `dotpacking8-scalar4bit` | `u16` |

`dotpacking8-scalar` is still accepted as the old name of `dotpacking8-scalaru8`. The list lives
in one place, `src/value_types.rs` (`VALUE_TYPES` and the `with_dataset_type!` macro), used by
`build_inverted_index`, `perf_inverted_index` and `full_scan_inverted_index`.
`full_scan_inverted_index` now takes any value type and a `--component-type` (default value type:
`dotpacking8-scalaru8`).

### Rust API: Seismic as a vectorium `Index`

- `InvertedIndexBase` and `SeismicIndex` implement vectorium's `Index` and `IndexStats` traits
  (`Query = SparseVectorView<C, f32>`, `Distance = DotProduct`), so a Seismic index can be used
  wherever a generic vectorium index is expected, e.g. as the first stage of kANNolo's
  `RerankIndex`. The query-time knobs are bundled in the new `seismic::SearchParams`
  (`query_cut`, `heap_factor`, `n_knn`, `first_sorted`; defaults match `perf_inverted_index`).
  Unlike the inherent `search`, the trait method sorts unsorted query components and drops those
  beyond the index dimensionality instead of panicking. Results carry internal vector ids;
  `SeismicIndex::remap_doc_ids` maps them to document ids. Contributed by Rohan Jha (#20).

### Experiments

- The experiment configs that used `dotvbyte` now use `dotpacking8-scalaru8`.
- `scripts/run_experiments.py` runs on Python 3.14 (it no longer uses `ir_measures.parse_measure`,
  broken there), and exits with code 1 when an experiment fails (it exited 0).

### Performance (MS MARCO cocondenser, against 0.5.2; 0.5.3 and 0.5.4 did not change the query code)

- Query time is unchanged or lower for `blocked-sparse` and the DotPacking8 encoders. This needed
  vectorium `eb4ca45`: the 256-byte prefetch cap inherited from vectorium `main` slowed
  `blocked-sparse` queries by up to 24% and DotPacking8 F16 by up to 9%; the cap is now 512 bytes.
- `dotpacking8-scalaru8` against the `dotvbyte` it replaces: same query time or faster, 4–8% less
  memory, recall slightly higher.
- DotPacking8 indexes build about 20% faster.
- Plain `f16` queries are 1–2% slower (cause not found).

## 0.5.4

### Longer token strings in the Python interface

- **Token dtype widened from `U30` to `U100`.** `get_seismic_string()` now
  returns `'U100'` and every token-string argument of the Python classes
  (document tokens, query components, query ids) expects that width. NumPy
  silently truncates strings that exceed the fixed width, so with `U30` two
  distinct terms sharing their first 30 characters (common for word-level
  encoders on corpora with long terms or URLs) collapsed onto the same token
  and `SeismicDataset.add_document` crashed with the opaque vectorium assert
  "Components must be given in strictly ascending order". Colliding terms are
  not merged on purpose: scores must match an exact dot product on the
  original terms. The wider dtype costs 400 bytes per element instead of 120,
  only for the Python-side arrays; the index stores integer ids and is
  unaffected.
  → `MAX_TOKEN_LEN`, `SEISMIC_STRING` in `src/pylib/mod.rs`

  **Breaking for callers that hardcode `'U30'`.** An array with any other
  dtype is rejected with a `TypeError` that names the argument, the dtype
  received and the one expected. Build arrays with
  `dtype=get_seismic_string()`, or convert existing ones with
  `.astype(get_seismic_string())`.

  **Indexes built with 0.5.3 or earlier.** Their token map holds the strings
  truncated to 30 characters. Loaded in 0.5.4, a query token longer than 30
  characters arrives in full, matches no entry and is silently dropped:
  lower recall, no error. Rebuild such indexes with 0.5.4, or truncate query
  tokens to 30 characters until then.

- **`add_document` raises `ValueError` on duplicate tokens** instead of
  tripping the vectorium assert. The message names the document id and the
  token, and adds the fixed-width truncation explanation when a token of the
  document fills the whole width. The check runs before anything is stored.
  Previously the rejected document's id was recorded without its vector, so a
  caller that caught the exception and kept adding documents got every later
  result labelled with the wrong document id. The Rust
  `SeismicDataset::add_document` now returns `Result<(), String>`.
  → `SeismicDataset::add_document` in `src/inverted_index_wrapper.rs`

- **No Rust panic on a wrong array dtype.** `batch_search` (index classes
  and datasets) extracted each per-query array with `unwrap()`, so a wrong
  dtype inside the list surfaced as a `PanicException`. It is now a
  `TypeError` that names the offending list entry.
  → `extract_tokens`, `extract_values` in `src/pylib/mod.rs`

## 0.5.2

### Faster index construction

Optimizations to the index build. None of them changes the built index: the
produced posting lists, blocks, and summaries are identical to the previous
implementation, up to the arbitrary choice among postings tied at the global
pruning threshold.

- **Parallel global-threshold pruning.** The global threshold is found by
  sorting all posting scores in parallel instead of scanning them through a
  serial heap; all postings scoring at or above the threshold are then
  distributed to components in parallel, and the surplus of scores tied at
  the threshold is dropped deterministically (thread-count-independently)
  from the tails of the posting lists. The old serial implementation is kept
  as a test oracle.
  → `global_threshold_pruning` in `src/inverted_index.rs`

- **Longest-first (LPT) posting-list dispatch.** Posting lists are built
  longest-first so a long list picked up late cannot leave the other workers
  idle; results are restored to component order afterwards.
  → the parallel loop in `InvertedIndexBase::build`, `src/inverted_index.rs`

- **Dense scratch for summaries.** The per-block `HashMap` used to accumulate
  component maxima is replaced by a dense epoch-stamped scratch table reused
  across all blocks of a posting list, and top-k selection uses
  `select_nth_unstable_by` instead of heap-based `k_largest_by`. Above 2^20
  components — where allocating the dense table for every posting list would
  dominate the build — the scratch falls back to an `FxHashMap`-backed slot
  table with the same layout, producing identical summaries.
  → `SummaryScratch`, `accumulate_block_max`, `fixed_size_summary`,
  `energy_preserving_summary` in `src/posting_list.rs`

- **CSR inverted index for the clustering.** The temporary centroid index used
  by the random k-means is now a flat offsets + postings layout (one contiguous
  array instead of one allocation per component), built with a counting sort,
  with scores converted to `f32` once at build time.
  → `FastInvertedIndex` in `src/utils.rs`

- **Survivors-only reassignment pass.** After dissolving too-small clusters,
  documents are reassigned against an index holding only the surviving
  centroids instead of the full one, shrinking the score accumulator, the
  posting traversal, and the argmax by the same factor. Assignments are
  bit-for-bit identical to the full-index version.
  → second pass of `do_random_kmeans_on_docids_ii_approx_dot_product` in
  `src/utils.rs`

- **Exact-dot-product k-means on the same index.** The exact variant of the
  clustering builds its pruned centroid index with `FastInvertedIndex` as well
  (score-based per-component pruning via `FastInvertedIndex::pruned`) instead
  of per-component vectors.
  → `do_random_kmeans_on_docids_ii_dot_product` in `src/utils.rs`
