#!/usr/bin/env bash
# Launch the LiLSR k=10 shrunk grid for summary_dotpacking8.
# summary_dotvbyte already completed (2026-05-26 07:40 run, 20/20 builds done).
# summary_f16 was intentionally skipped — its best_configs/ was extracted
# from the pooled explorative + highmem data, no re-run needed.
set -euo pipefail

cd /home/local/ADUNIPI/a050143/seismic_private

echo "[$(date -u +%FT%TZ)] starting summary_dotpacking8 grid"
python scripts/run_grid_search.py --exp experiments/ICDE2026/msmarco-v1-lilsr/k_10/summary_dotpacking8/grid_search.toml

echo "[$(date -u +%FT%TZ)] done"
