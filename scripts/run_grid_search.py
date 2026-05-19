import argparse
import sys
import os
import subprocess
import re
import glob
import hashlib

import json
from datetime import datetime
from termcolor import colored

import itertools

# from run_experiments import
from run_experiments import (
    compile_rust_code,
    run_experiment,
    parse_toml,
)

def hash_params(params_dict):
    """Deterministic hash of a parameter combination."""
    canonical = json.dumps(params_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def scan_completed_combinations(grid_folder, expected_query_count):
    """Scan a grid search directory for completed combinations.

    Returns a dict mapping param_hash -> index for completed combinations.
    """
    completed = {}
    json_files = glob.glob(os.path.join(grid_folder, "building_combination_*.json"))

    for json_path in json_files:
        basename = os.path.basename(json_path)
        match = re.match(r"building_combination_(\d+)\.json", basename)
        if not match:
            continue
        idx = int(match.group(1))

        with open(json_path) as f:
            params = json.load(f)
        params = {k: v for k, v in params.items() if not k.startswith("__")}
        param_hash = hash_params(params)

        exp_dirs = glob.glob(os.path.join(grid_folder, f"building_combination_{idx}_*"))
        exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]

        for exp_dir in exp_dirs:
            report_path = os.path.join(exp_dir, "report.tsv")
            if os.path.exists(report_path):
                with open(report_path) as f:
                    row_count = sum(1 for line in f if line.strip()) - 1
                if row_count >= expected_query_count:
                    completed[param_hash] = idx
                    break

    return completed


def generate_indexing_parameters_combinations(params):
    # Extract keys and values from the dictionary
    keys, values = zip(*params.items())
    # Generate all possible combinations
    all_combinations = itertools.product(*values)
    
    unique_combinations = set()
    for combination in all_combinations:
        combo_dict = dict(zip(keys, combination))
        
        # If 'clustering-algorithm' is 'random-kmeans', set 'kmeans-doc-cut' and 'kmeans-pruning-factor' to a fake value. 
        # These parameters are not needed.
        if combo_dict["clustering-algorithm"] == "random-kmeans":
            combo_dict["kmeans-doc-cut"] = 0
            combo_dict["kmeans-pruning-factor"] = 0.0

        # If 'clustering-algorithm' is 'random-kmeans-inverted-index-approx', set 'kmeans-pruning-factor' to a fake value. 
        # This parameter is not needed.
        if combo_dict["clustering-algorithm"] == "random-kmeans-inverted-index-approx":
            combo_dict["kmeans-pruning-factor"] = 0.0
        
        # Convert dict to tuple of sorted items for deduplication and add to set
        combo_tuple = tuple(sorted(combo_dict.items()))
        unique_combinations.add(combo_tuple)
    
    # Convert back to list of dictionaries
    return [dict(combo) for combo in unique_combinations]

def generate_query_combinations(params):
    keys, values = zip(*params.items())
    all_combinations = itertools.product(*values)
    
    combination_dict = {}
    for i, combination in enumerate(all_combinations, start=1):
        combo_key = f"combination_{i}"
        combo_dict = dict(zip(keys, combination))
        combination_dict[combo_key] = combo_dict
    
    return combination_dict


def compute_dataset_space_usage(config_data, grid_folder):
    """Run dataset_space_usage binary to log the plain forward index size."""
    for k, v in config_data["folder"].items():
        if v.startswith("~"):
            config_data["folder"][k] = os.path.expanduser(v)

    dataset_path = os.path.join(
        config_data["folder"]["data"], config_data["filename"]["dataset"]
    )
    output_tsv = os.path.join(grid_folder, "space_usage.tsv")
    binary = "./target/release/dataset_space_usage"

    command = f"{binary} --input-file {dataset_path} --output-file {output_tsv}"

    print()
    print(colored("Computing dataset space usage:", "green"))
    print(colored("Command:", "blue"), command)

    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=None
    )
    print(result.stdout.decode(), end="")

    if result.returncode != 0:
        print(colored("ERROR: dataset_space_usage failed!", "red"))
        sys.exit(1)

    print(colored(f"Space usage written to {output_tsv}", "yellow"))


def main(experiment_config_filename, resume_path=None):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print("Error: Configuration data is empty.")
        sys.exit(1)

    # Get the experiment name from the configuration
    grid_name = config_data.get("name")
    print(f"Running Grid: {grid_name}")

    # Directory selection: resume existing or create new
    if resume_path:
        grid_folder = resume_path
        if not os.path.isdir(grid_folder):
            print(f"Error: Resume directory does not exist: {grid_folder}")
            sys.exit(1)
        print(f"Resuming grid search in: {grid_folder}")
    else:
        timestamp = str(datetime.now()).replace(" ", "_")
        grid_folder = os.path.join(
            config_data["folder"]["experiment"], f"{grid_name}_{timestamp}"
        )
        os.makedirs(grid_folder, exist_ok=True)

    # Compile first; skip space usage when resuming (already computed)
    compile_rust_code(config_data, grid_folder)
    if not resume_path:
        compute_dataset_space_usage(config_data, grid_folder)

    print()
    print(colored("Grid search information:", "yellow"))

    print(json.dumps(config_data["indexing_parameters"], indent=4))

    query_combinations = generate_query_combinations(config_data["querying_parameters"])
    expected_query_count = len(query_combinations)

    # Scan for completed combinations if resuming
    completed_hashes = {}
    next_index = 0

    if resume_path:
        completed_hashes = scan_completed_combinations(grid_folder, expected_query_count)
        if completed_hashes:
            next_index = max(completed_hashes.values()) + 1
        print(colored(f"Found {len(completed_hashes)} completed combination(s), "
                      f"next index: {next_index}", "green"))

    combinations = generate_indexing_parameters_combinations(config_data["indexing_parameters"])

    # Warn about orphaned completions (completed but not in current grid)
    if resume_path:
        current_hashes = {hash_params(c) for c in combinations}
        orphaned = set(completed_hashes.keys()) - current_hashes
        if orphaned:
            print(colored(f"Warning: {len(orphaned)} previously completed combination(s) "
                          f"are not in the current parameter grid (config may have changed)", "yellow"))

    print(f"Total combinations: {len(combinations)}, "
          f"already completed: {len(completed_hashes)}")
    print("Run an experiment for each building configuration")

    skipped = 0
    for building_config in combinations:
        param_hash = hash_params(building_config)

        if param_hash in completed_hashes:
            skipped += 1
            print(f"\nSkipping completed combination (hash={param_hash})")
            continue

        i = next_index
        next_index += 1

        print()
        print(f"Running building combination {i} "
              f"(hash={param_hash}, {skipped} skipped)")
        print(f"Running building combination {i} with {json.dumps(building_config, indent=4)}")

        experiment_config = {}
        experiment_config["folder"] = config_data["folder"]
        experiment_config["folder"]["experiment"] = grid_folder

        experiment_config["filename"] = config_data["filename"]
        experiment_config["settings"] = config_data["settings"]

        experiment_config["name"] = f"building_combination_{i}"

        experiment_config["query"] = query_combinations

        # Copy build/compile commands if they exist
        if "compile-command" in config_data:
            experiment_config["compile-command"] = config_data["compile-command"]
        if "build-command" in config_data:
            experiment_config["build-command"] = config_data["build-command"]
        if "query-command" in config_data:
            experiment_config["query-command"] = config_data["query-command"]

        experiment_config["indexing_parameters"] = building_config
        with open(os.path.join(grid_folder, f"building_combination_{i}.json"), "w") as f:
            json.dump(building_config, f, indent=4)
        try:
            run_experiment(experiment_config)
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid search of seismic experiments on a dataset and find the best configurations to query it."
    )
    parser.add_argument(
        "--exp", required=True, help="Path to the grid configuration TOML file."
    )
    parser.add_argument(
        "--resume", required=False, default=None,
        help="Path to an existing grid search directory to resume."
    )
    args = parser.parse_args()

    main(args.exp, resume_path=args.resume)
