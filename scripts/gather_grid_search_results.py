"""
This script collects all grid search results into a single final_report.tsv file.

It is meant to be run on a main grid search folder (e.g., ./grid_search), where each subfolder
contains one or more index build results in subdirectories named building_combination_*/.
Each of these contains a report.tsv file (with performance metrics) and an experiment_config.toml
file (with configuration parameters).

For each combination in each report.tsv file, this script extracts:
- The indexing and settings parameters from the TOML
- The specific query.combination_z parameters from the TOML
- The associated metrics from the report.tsv row

The final report is saved as final_report.tsv in the main folder.

Note:
Grid search can be executed using script/run_grid_search.py with parameters specified in a TOML file.
An example TOML configuration is provided in:
    experiments/grid_searches/example_grid_search.toml
"""

import os
import glob
import toml
import pandas as pd

def gather_grid_search_results(main_grid_search_folder):
    final_rows = []

    # Each subfolder is a grid search root
    for subdir in sorted(os.listdir(main_grid_search_folder)):
        full_subdir_path = os.path.join(main_grid_search_folder, subdir)
        if not os.path.isdir(full_subdir_path):
            continue

        build_folders = glob.glob(os.path.join(full_subdir_path, "building_combination_*"))
        build_folders = [f for f in build_folders if os.path.isdir(f)]
        for build_folder in sorted(build_folders):
            config_path = os.path.join(build_folder, "experiment_config.toml")
            report_path = os.path.join(build_folder, "report.tsv")

            if not os.path.exists(config_path):
                print(f"❌ Skipping {build_folder} – missing file: experiment_config.toml")
                continue
            if not os.path.exists(report_path):
                print(f"❌ Skipping {build_folder} – missing file: report.tsv")
                continue

            try:
                config = toml.load(config_path)
            except Exception as e:
                print(f"❌ Failed to load {config_path}: {e}")
                continue

            index_params = config.get("indexing_parameters", {})
            settings = config.get("settings", {})
            # Access the [query] section as a map
            query_section = config.get("query", {})

            try:
                report_df = pd.read_csv(report_path, sep="\t")
            except Exception as e:
                print(f"❌ Failed to read {report_path}: {e}")
                continue

            for _, row in report_df.iterrows():
                subsection = row["Subsection"]
                try:
                    combination_id = int(subsection.split("_")[-1])
                except Exception as e:
                    print(f"⚠️  Skipping row with invalid Subsection: {subsection} ({e})")
                    continue

                query_key = f"combination_{combination_id}"
                if query_key not in query_section:
                    print(f"⚠️  No query parameters found for [query.{query_key}] in {config_path}")
                query_info = query_section.get(query_key, {})

                final_row = {
                    "experiment": subdir,
                    "build_folder": os.path.basename(build_folder),
                    "combination_id": combination_id,
                }

                final_row.update({f"index.{k}": v for k, v in index_params.items()})
                final_row.update({f"settings.{k}": v for k, v in settings.items()})
                # Always add query.combination_z parameters, even if empty
                for k in query_info:
                    final_row[f"query.{k}"] = query_info[k]
                final_row.update(row.to_dict())

                final_rows.append(final_row)

    if not final_rows:
        print("⚠️  No valid experiments found. No output file generated.")
        return

    df = pd.DataFrame(final_rows)
    output_path = os.path.join(main_grid_search_folder, "final_report.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    print(f"✅ Final report saved to: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python gather_grid_search_results.py <main_grid_search_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    gather_grid_search_results(folder)