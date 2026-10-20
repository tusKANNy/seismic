# Changes

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
  with one step per component).
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

### Experiments

- The experiment configs that used `dotvbyte` now use `dotpacking8-scalaru8`.
- `scripts/run_experiments.py` runs on Python 3.14 (it no longer uses `ir_measures.parse_measure`,
  broken there), and exits with code 1 when an experiment fails (it exited 0).
- `experiments/vectorium_compression/` holds the before/after evaluation (`REPORT.md`) and smoke
  configs covering every (component type, value type) pair.

### Performance (MS MARCO cocondenser, against 0.5.2)

- Query time is unchanged or lower for `blocked-sparse` and the DotPacking8 encoders. This needed
  vectorium `eb4ca45`: the 256-byte prefetch cap inherited from vectorium `main` slowed
  `blocked-sparse` queries by up to 24% and DotPacking8 F16 by up to 9%; the cap is now 512 bytes.
- `dotpacking8-scalaru8` against the `dotvbyte` it replaces: same query time or faster, 4–8% less
  memory, recall slightly higher.
- DotPacking8 indexes build about 20% faster.
- Plain `f16` queries are 1–2% slower (cause not found).

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
