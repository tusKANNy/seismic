import re 
import os
import sys
import time
import socket
import argparse
import subprocess
from datetime import datetime

import ir_measures
import toml
import psutil

import numpy as np
import pandas as pd

from termcolor import colored

def parse_toml(filename):
    """Parse the TOML configuration file."""
    try:
        return toml.load(filename)
    except Exception as e:
        print(f"Error reading the TOML file: {e}")
        return None


def get_git_info(experiment_dir):
    """Get Git repository information and save it to git.output."""
    print()
    print(colored("Git info", "green"))
    git_output_file = os.path.join(experiment_dir, "git.output")

    try:
        with open(git_output_file, "w") as git_output:
            # Get current branch
            branch_process = subprocess.Popen("git rev-parse --abbrev-ref HEAD", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            branch_name = branch_process.stdout.read().decode().strip()
            branch_process.wait()

            # Get current commit id
            commit_process = subprocess.Popen("git rev-parse HEAD", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commit_id = commit_process.stdout.read().decode().strip()
            commit_process.wait()

            # Write to git.output
            git_output.write(f"Current Branch: {branch_name}\n")
            git_output.write(f"Commit ID: {commit_id}\n")
            print(f"Current Branch: {branch_name}")
            print(f"Commit ID: {commit_id}")

    except Exception as e:
        print("An error occurred while retrieving Git information:", e)
        sys.exit(1)


def compile_rust_code(configs, experiment_dir):
    """Compile the Rust code and save output."""
    print()
    print(colored("Compiling the Rust code", "green"))
    
    compile_command = configs.get("compile-command", "RUSTFLAGS='-C target-cpu=native' cargo build --release --bin build_inverted_index --bin perf_inverted_index --bin dataset_space_usage --no-default-features")

    compilation_output_file = os.path.join(experiment_dir, "compiler.output")

    try:
        print("Compiling Rust code with", compile_command)
        with open(compilation_output_file, "w") as comp_output:
            compile_process = subprocess.Popen(compile_command, shell=True, stdout=subprocess.PIPE, stderr=None)
            for line in iter(compile_process.stdout.readline, b''):
                decoded_line = line.decode()
                print(decoded_line, end='')  # Print each line as it is produced
                comp_output.write(decoded_line)  # Write each line to the output file
            compile_process.stdout.close()
            compile_process.wait()

        if compile_process.returncode != 0:
            print("Rust compilation failed.")
            sys.exit(1)
        print("Rust code compiled successfully.")

    except Exception as e:
        print()
        print(colored("ERROR: Problems during Rust compilation:", "red"), e)
        sys.exit(1)

def shrink_name(original_name):
    
    replacement_dict = {
        "clustering-algorithm": "c-a", 
        "centroid-fraction": "c-f", 
        "kmeans-pruning-factor": "k-p-f", 
        "kmeans-doc-cut": "k-d-c", 
        "pruning-strategy": "p-s",
        "max-fraction": "m-f",
        "min-cluster-size": "m-c-s",
        "global-threshold": "g-t",
        "summary-energy": "s-e",
        "n-postings": "n-p",
        "component-type": "c-t",
        "value-type": "v-t",
        "random-kmeans-inverted-index-approx": "r-k-i-a",
        "pruning-strategy": "p-s"
    }
    
    final_name = original_name
    for k,v in replacement_dict.items():
        final_name = final_name.replace(k, v)
    return final_name

def get_index_filename(base_filename, configs):
    """Generate the index filename based on the provided parameters."""
    name = [
        base_filename, 
    ] + sorted(f"{k}_{v}" for k, v in configs["indexing_parameters"].items())
    
    join_name = "_".join(str(l) for l in name)
    #print(f"ORIGINAL NAME {join_name}")
    if len(join_name) > 240:
        join_name = shrink_name(join_name)
    #print(f"SHRINKED NAME {join_name}")
    
    return join_name


def build_index(configs, experiment_dir):
    build_command = configs.get("build-command", "./target/release/build_inverted_index")
    return build_base_index(configs, experiment_dir, build_command=build_command)


def append_value_type(command_and_params, configs):
    value_type = configs['indexing_parameters']["value-type"]
    valid_value_types = {"f16", "bf16", "f32", "fixedu8", "fixedu16", "dotvbyte", "dotvbyte-scalar", "dotvbyte-optimistic", "dotvbyte-optimistic-scalar", "dotpacking8-fixedu8", "dotpacking8-scalar", "dotpacking8-sparse-scalar", "dotpacking8-scalar4bit", "dotpacking8-centroid4bit", "dotpacking8-optimistic-scalar", "dotpacking8-f16", "dotpackingsparse-f16", "blocked-sparse"}
    if value_type not in valid_value_types:
        print(colored(f"ERROR: Invalid value-type '{value_type}'. Valid options are: {', '.join(valid_value_types)}", "red"))
        sys.exit(1)

    command_and_params.append(f"--value-type {value_type}")



        

def build_base_index(configs, experiment_dir, build_command):
    """Build the index using the provided configuration."""
    
    input_file =  os.path.join(configs["folder"]["data"], configs["filename"]["dataset"])
    index_folder = configs["folder"]["index"]

    os.makedirs(index_folder, exist_ok=True)
    output_file = os.path.join(index_folder, get_index_filename(configs["filename"]["index"], configs))
    
    print()
    print(colored("Dataset filename:", "blue"), input_file)
    print(colored("Index filename:", "blue"), output_file)

    #build_command = configs.get("build-command", "./target/release/build_inverted_index")

    command_and_params = [
        build_command,
        f"--input-file {input_file}",
        f"--output-file {output_file}",
        f"--n-postings {configs['indexing_parameters']['n-postings']}",
        f"--centroid-fraction {configs['indexing_parameters']['centroid-fraction']}",
        f"--knn {configs['indexing_parameters']['knn']}",
        f"--clustering-algorithm {configs['indexing_parameters']['clustering-algorithm']}",
    ] 

    if configs["filename"].get("knn_path", None):
        knn_path = os.path.join(configs['folder']['data'], configs['filename']['knn_path'])
        knn_path_arg = f"--knn-path {knn_path}"
        command_and_params.append(knn_path_arg)

    if configs['indexing_parameters'].get('summary-energy', None):
        command_and_params.append(f"--summary-energy {configs['indexing_parameters']['summary-energy']}")

    if configs['indexing_parameters'].get('kmeans-pruning-factor', None):
        kmeans_pruning_factor = configs['indexing_parameters']['kmeans-pruning-factor']
        command_and_params.append(f"--kmeans-pruning-factor {kmeans_pruning_factor}")

    if configs['indexing_parameters'].get('kmeans-doc-cut', None):
        kmeans_doc_cut = configs['indexing_parameters']['kmeans-doc-cut']
        command_and_params.append(f"--kmeans-doc-cut {kmeans_doc_cut}")

    if configs['indexing_parameters'].get('min-cluster-size', None):
        min_cluster_size = configs['indexing_parameters']['min-cluster-size']
        command_and_params.append(f"--min-cluster-size {min_cluster_size}")

    if configs['indexing_parameters'].get("max-fraction", None):
        max_fraction = configs['indexing_parameters']["max-fraction"]
        command_and_params.append(f"--max-fraction {max_fraction}")

    if configs['indexing_parameters'].get("alpha", None):
        alpha = configs['indexing_parameters']["alpha"]
        command_and_params.append(f"--alpha {alpha}")

    if configs['indexing_parameters'].get("component-type", None):
        component_type = configs['indexing_parameters']["component-type"]
        command_and_params.append(f"--component-type {component_type}")

    


    if configs['indexing_parameters'].get("value-type", None):
        append_value_type(command_and_params, configs)


    pruning_strategy = configs['indexing_parameters'].get("pruning-strategy", "global-threshold")
    command_and_params.append(f"--pruning-strategy {pruning_strategy}")

    if configs['indexing_parameters'].get("summarization-strategy", None):
        summarization_strategy = configs['indexing_parameters']["summarization-strategy"]
        command_and_params.append(f"--summarization-strategy {summarization_strategy}")

    command = ' '.join(command_and_params)

    # Print the command that will be executed
    print()
    print(colored(f"Indexing", "green"))
    print(colored(f"Indexing command:", "blue"), command)

    building_output_file = os.path.join(experiment_dir, "building.output")

    # Build the index and display output in real-time
    #print("Dataset summary")
    print()
    print("Some statistics of the dataset...")
    building_time = 0
    with open(building_output_file, "w") as build_output:
        build_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=None)
        for line in iter(build_process.stdout.readline, b''):
            decoded_line = line.decode()
            print(decoded_line, end='')  # Print each line as it is produced
            build_output.write(decoded_line)  # Write each line to the output file
            match = re.search(r"Time to build (\d+) secs", decoded_line)
            if match:
                building_time = int(match.group(1))
        build_process.stdout.close()
        build_process.wait()

    if build_process.returncode != 0:
        print(colored("ERROR: Indexing failed!", "red"))
        sys.exit(1)

    print(colored(f"Index built successfully in {building_time} secs!", "yellow"))
    return building_time, output_file+".index.seismic"

def compute_metric(configs, output_file, gt_file, metric):    
    qrels_path = configs['folder']['qrels_path']
    
    # Skip metric computation if qrels_path is empty or doesn't exist
    if not qrels_path or qrels_path.strip() == "" or not os.path.exists(qrels_path):
        print(f"Skipping metric computation: qrels_path is empty or file doesn't exist: '{qrels_path}'")
        return 0.0
    
    column_names = ["query_id", "doc_id", "rank", "score"]
    gt_pd = pd.read_csv(gt_file, sep='\t', names=column_names)
    res_pd = pd.read_csv(output_file, sep='\t', names=column_names)
    
    query_ids_path = os.path.join(configs['folder']['data'], configs['filename']['query_ids'])
    queries_ids = np.load(query_ids_path, allow_pickle=True)

    document_ids_path = os.path.join(configs['folder']['data'], configs['filename']['doc_ids'])
    doc_ids = np.load(os.path.realpath(document_ids_path), allow_pickle=True)
    
    gt_pd['query_id'] = gt_pd['query_id'].apply(lambda x: queries_ids[x])
    res_pd['query_id'] = res_pd['query_id'].apply(lambda x: queries_ids[x])
    
    gt_pd['doc_id'] = gt_pd['doc_id'].apply(lambda x: doc_ids[x])
    res_pd['doc_id'] = res_pd['doc_id'].apply(lambda x: doc_ids[x])
    
    df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "useless", "doc_id", "relevance"])
    #if "nq" in configs['name']: # the order of the fields in nq is different. 
    if len(pd.unique(df_qrels['useless'])) != 1:
        df_qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "doc_id", "relevance", "useless"])

    gt_pd['doc_id'] = gt_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    res_pd['doc_id'] = res_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    
    gt_pd['query_id'] = gt_pd['query_id'].astype(df_qrels.query_id.dtype)
    res_pd['query_id'] = res_pd['query_id'].astype(df_qrels.query_id.dtype)
    
    ir_metric = ir_measures.parse_measure(metric)
    
    metric_val = ir_measures.calc_aggregate([ir_metric], df_qrels, res_pd)[ir_metric]
    metric_gt = ir_measures.calc_aggregate([ir_metric], df_qrels, gt_pd)[ir_metric]
    
    print(f"Metric of the run ({ir_metric}): {round(metric_val, 4)}")
    print(f"Metric of the ground truth ({ir_metric}): {round(metric_gt, 4)}")
    return metric_val


def compute_accuracy(query_file, gt_file):
    column_names = ["query_id", "doc_id", "rank", "score"]
    gt_pd = pd.read_csv(gt_file, sep='\t', names=column_names)
    res_pd = pd.read_csv(query_file, sep='\t', names=column_names)

    # Group both dataframes by 'query_id' and get unique 'doc_id' sets
    gt_pd_groups = gt_pd.groupby('query_id')['doc_id'].apply(set)
    res_pd_groups = res_pd.groupby('query_id')['doc_id'].apply(set)

    # Compute the intersection size for each query_id in both dataframes
    intersections_size = {
        query_id: len(gt_pd_groups[query_id] & res_pd_groups[query_id]) if query_id in res_pd_groups else 0
        for query_id in gt_pd_groups.index
    }

    # Computes total number of results in the ground truth
    total_results = len(gt_pd)
    total_intersections = sum(intersections_size.values())
    
    accuracy = total_intersections/total_results
    
    print(f"Accuracy: {round(accuracy, 4)}")
    return accuracy


def query_execution(configs, query_config, experiment_dir, subsection_name, index_file):
    """Execute a query based on the provided configuration."""
    #index_file = os.path.join(configs["folder"]["index"], get_index_filename(configs["filename"]["index"], configs))
    query_file =  os.path.join(configs["folder"]["data"], configs["filename"]["queries"] ) 
    
    output_file = os.path.join(experiment_dir, f"results_{subsection_name}")
    log_output_file =  os.path.join(experiment_dir, f"log_{subsection_name}") 

    query_command = configs.get("query-command", "./target/release/perf_inverted_index")

    command_and_params = [
        configs['settings']['NUMA'] if "NUMA" in configs['settings'] else "",
        query_command, 
        f"--index-file {index_file}",
        f"-k {configs['settings']['k']}",
        f"--query-file {query_file}",
        f"--query-cut {query_config['query-cut']}",
        f"--heap-factor {query_config['heap-factor']}",
        f"--n-runs {configs['settings']['n-runs']}",
        f"--output-path {output_file}",
          
    ]

    if "knn" in query_config:
        command_and_params.append(f"--n-knn {query_config['knn']}" )
    
    if query_config.get("first_sorted", False):
        print(colored("You are using the 'first_sorted' parameter, which is deprecated. Please, use 'first-sorted' instead.", "red"))
        command_and_params.append("--first-sorted")
    
    if query_config.get("first-sorted", False):
        command_and_params.append("--first-sorted")
        
    

    if configs['indexing_parameters'].get("component-type", None):
        component_type = configs['indexing_parameters']["component-type"]
        command_and_params.append(f"--component-type {component_type}")

    if configs['indexing_parameters'].get("value-type", None):
        value_type = configs['indexing_parameters']["value-type"]
        valid_value_types = {"f16", "bf16", "f32", "fixedu8", "fixedu16", "dotvbyte", "dotvbyte-scalar", "dotvbyte-optimistic", "dotvbyte-optimistic-scalar", "dotpacking8-fixedu8", "dotpacking8-scalar", "dotpacking8-sparse-scalar", "dotpacking8-scalar4bit", "dotpacking8-centroid4bit", "dotpacking8-optimistic-scalar", "dotpacking8-f16", "dotpackingsparse-f16", "blocked-sparse"}
        if value_type not in valid_value_types:
            print(colored(f"ERROR: Invalid value-type '{value_type}'. Valid options are: {', '.join(valid_value_types)}", "red"))
            sys.exit(1)
        command_and_params.append(f"--value-type {value_type}")

    command = " ".join(command_and_params)

    print(f"Executing query for subsection '{subsection_name}'")
    print(colored(f"Query command: ", "blue"), command.strip())

    pattern = r"\tTotal: (\d+) Bytes" # Pattern to match the total memory usage

    query_time = 0
    # Run the query and display output in real-time
    print(f"Running query for subsection: {subsection_name}...")
    with open(log_output_file, "w") as log:
        query_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=None)
        for line in iter(query_process.stdout.readline, b''):
            decoded_line = line.decode()
            if decoded_line.startswith("Time ") and decoded_line.strip().endswith("microsecs per query"):
                query_time = int(decoded_line.split()[1])

            match = re.search(pattern, decoded_line)
            if match:
                memory_usage = int(match.group(1))
            print(decoded_line, end='')  # Print each line as it is produced
            log.write(decoded_line)  # Write each line to the output file
        query_process.stdout.close()
        query_process.wait()

    if query_process.returncode != 0:
        print(f"Query execution for subsection '{subsection_name}' failed.")
        sys.exit(1)

    print(f"Query for subsection '{subsection_name}' executed successfully.")

    gt_file = os.path.join(configs['folder']['data'], configs['filename']['groundtruth'])
    metric = configs['settings']['metric']
    return query_time, compute_accuracy(output_file, gt_file), compute_metric(configs, output_file, gt_file, metric), memory_usage


def get_machine_info(configs, experiment_folder):

    return
    machine_info_file = os.path.join(experiment_folder, "machine.output")
    machine_info = open(machine_info_file, "w")

    date = datetime.now()
    machine = socket.gethostname()
    cpu = psutil.cpu_percent(interval=1)
    
    memory_free = psutil.virtual_memory().free // (1024 ** 3)
    memory_avail = psutil.virtual_memory().available // (1024 ** 3)
    memory_total = psutil.virtual_memory().total // (1024 ** 3)
    
    load = str(psutil.getloadavg())[1:-1]
    num_cpus = psutil.cpu_count()
    
    machine_info.write(f"----------------------\n")
    machine_info.write(f"Hardware configuration\n")
    machine_info.write(f"----------------------\n")
    machine_info.write(f"Date: {date}\n")
    machine_info.write(f"Machine: {machine}\n")
    machine_info.write(f"CPU usage (%): {cpu}\n")
    machine_info.write(f"Machine load: {load}\n")
    machine_info.write(f"Memory (free, GiB): {memory_free}\n")
    machine_info.write(f"Memory (avail, GiB): {memory_avail}\n")
    machine_info.write(f"Memory (total, GiB): {memory_total}\n")
    
    print()
    print(colored("Hardware configuration", "green"))
    print(f"Date: {date}")
    print(f"Machine: {machine}")
    print(f"CPU usage (%): {cpu}")
    print(f"Machine load: {load}")
    print(f"Memory (free, GiB): {memory_free}")
    print(f"Memory (avail, GiB): {memory_avail}")
    print(f"Memory (total, GiB): {memory_total}")
    print(f"for detailed information, check the hardware log file: {machine_info_file}")

    machine_info.write(f"\n---------------------\n")
    machine_info.write(f"cpufreq configuration\n")
    machine_info.write(f"---------------------\n")

    command_governor = 'cpufreq-info | grep "performance" | grep -v "available" | wc -l'
    governor = subprocess.Popen(command_governor, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    governor.wait()

    for line in iter(governor.stdout.readline, b''):
        cpus_with_performance_governor = int(line.decode())
        machine_info.write(f'Number of CPUs with governor set to "performance" (should be equal to the number of CPUs below): {cpus_with_performance_governor}\n')

    # checking if the hardware looks well configured...
    if (num_cpus != cpus_with_performance_governor):
        print()
        print(colored("ERROR: Problems with hardware configuration found!", "red"))
        print(colored("Your CPU is not set to performance mode. Please, run `cpufreq-info` for more details.", "red"))
        print()

    machine_info.write(f"\n-----------------\n")
    machine_info.write(f"CPU configuration\n")
    machine_info.write(f"-----------------\n")

    command_cpu = 'lscpu'
    cpu = subprocess.Popen(command_cpu, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cpu.wait()

    for line in iter(cpu.stdout.readline, b''):
        decoded_line = line.decode()
        machine_info.write(decoded_line)

    if ("NUMA" in configs['settings']):
        machine_info.write(f"\n------------------------------------------------------------------------------\n")
        machine_info.write(f"NUMA execution command (check if CPU IDs corresponds to physical ones (no HT))\n")
        machine_info.write(f"------------------------------------------------------------------------------\n")
        machine_info.write(f'Shell command: "{configs["settings"]["NUMA"]}"\n')

        machine_info.write(f"\n------------------\n")
        machine_info.write(f"NUMA configuration\n")
        machine_info.write(f"------------------\n")

        command_numa = 'numactl --hardware'
        numa = subprocess.Popen(command_numa, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        numa.wait()

        for line in iter(numa.stdout.readline, b''):
            decoded_line = line.decode()
            machine_info.write(decoded_line)

    machine_info.close()
    return


#def remove_index_files(configs):
def remove_index_files(index_file_path):
    
    try:
        if os.path.exists(index_file_path):
            os.remove(index_file_path)
            print(colored(f"Index file removed: {index_file_path}", "yellow"))
        else:
            print(colored(f"Index file not found (already removed?): {index_file_path}", "yellow"))
    except Exception as e:
        print(colored(f"Warning: Could not remove index file {index_file_path}: {e}", "red"))


def run_experiment(config_data):
    """Run the seismic experiment based on the provided configuration."""

     # Get the experiment name from the configuration
    experiment_name = config_data.get("name")
    print(colored(f"Running experiment:", "blue"), experiment_name)

    for k, v in config_data["folder"].items():
        if v.startswith("~"):
            v = os.path.expanduser(v)
            config_data["folder"][k] = v

   #print(config_data)

    # Create an experiment folder with date and hour
    timestamp  = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    experiment_folder = os.path.join(config_data["folder"]["experiment"], f"{experiment_name}_{timestamp}")

    os.makedirs(experiment_folder, exist_ok=True)

    # Dump)the configuration settings to a TOML file
    with open(os.path.join(experiment_folder, "experiment_config.toml"), 'w') as report_file:
        report_file.write(toml.dumps(config_data))

    # Retrieving hardware information
    get_machine_info(config_data, experiment_folder)

    # Store the output of the Rust compilation and index building processes
    get_git_info(experiment_folder)
    


    compile_rust_code(config_data, experiment_folder)

    building_time = 0
    if config_data['settings']['build']:
        building_time, index_path = build_index(config_data, experiment_folder)
    else:
        print("Index is already built!")
        # Reconstruct the index path
        index_folder = config_data["folder"]["index"]
        index_filename = get_index_filename(config_data["filename"]["index"], config_data)
        index_path = os.path.join(index_folder, index_filename) + ".index.seismic"
        print(f"Using existing index: {index_path}")

    metric = config_data['settings']['metric']

    print()
    print(colored(f"Evaluation", "green"))
    print(f"Evaluation runs with metric {metric}")

    # Execute queries for each subsection under [query]
    with open(os.path.join(experiment_folder, "report.tsv"), 'w') as report_file:
        report_file.write(f"Subsection\tQuery Time (microsecs)\tRecall\t{metric}\tMemory Usage (Bytes)\tBuilding Time (secs)\n")
        if 'query' in config_data:
            for subsection, query_config in config_data['query'].items():
                query_time, recall, metric, memory_usage = query_execution(config_data, query_config, experiment_folder, subsection, index_path)
                report_file.write(f"{subsection}\t{query_time}\t{recall}\t{metric}\t{memory_usage}\t{building_time}\n")

    # Remove index files if delete parameter is set to true
    if config_data['settings'].get('delete', False):
        remove_index_files(index_path)

def main(experiment_config_filename):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print()
        print(colored("ERROR: Configuration data is empty.", "red"))
        sys.exit(1)
    
    try:
        run_experiment(config_data)
    except Exception as e:
        print()
        print(colored("ERROR: Experiment execution failed!", "red"))
        print(colored(f"Error details: {str(e)}", "red"))
        print(colored(f"Error type: {type(e).__name__}", "yellow"))
        
        # Print traceback for debugging
        import traceback
        print(colored("Full traceback:", "yellow"))
        traceback.print_exc()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a seismic experiment on a dataset and query it.")
    parser.add_argument("--exp", required=True, help="Path to the experiment configuration TOML file.")
    args = parser.parse_args()

    main(args.exp)
    sys.exit(0)
