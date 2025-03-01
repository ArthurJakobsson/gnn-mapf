# Standard library imports
import os
import subprocess
import argparse
import shutil
import multiprocessing
import datetime
import numpy as np

# Custom imports
from custom_utils.common_helper import str2bool
from sbatch_master_process_runner import generate_sh_script, startup_small
from custom_utils.argparser_main import parse_arguments

# Global variable to track timing between events
last_recorded_time = datetime.datetime.now()

def log_time(event_name):
    """
    Logs timing information for different events to a file
    Args:
        event_name: Name of the event being timed
    """
    cur_time = datetime.datetime.now()
    with open(f"./{LE}/timing.txt", mode='a') as file:
        file.write(f"{event_name} recorded at {cur_time}. \t\t Duration: \t {(cur_time-last_recorded_time).total_seconds()} \n")

if __name__ == "__main__":
    flags = [
        "expnum",
        "mini_test",
        "numScensToCreate",
        "num_parallel",
        "data_folder",
        "k",
        "m",
        "lr",
        "batch_size",
        "relu_type",
        "expName",
        "numAgents",
        "extra_layers",
        "bd_pred",
        "which_setting",
        "percent_for_succ",
        "which_section",
        "iternum",
        "timeLimit",
        "num_scens",
        "suboptimality",
        "dataset_size"
    ]

    args = parse_arguments(flags)

    # Set conda environment based on execution setting
    if args.which_setting == "Arthur":
        conda_env = None
    elif args.which_setting == "Rishi":
        conda_env = "pytorchfun"
    elif args.which_setting == "PSC":
        pass
    else:
        raise ValueError(f"Invalid setting: {args.which_setting}")

    # Handle path configurations for numAgents json files
    if ".json" in args.numAgents and "map_configs" not in args.numAgents:
        args.numAgents = "map_configs/"+args.numAgents

    # Set source directory for maps and scenarios
    if args.mini_test:
        source_maps_scens = f"./data_collection/data/{args.data_folder}"
    else:
        source_maps_scens = "./data_collection/data/benchmark_data"

    if args.num_scens != 0:
        source_maps_scens = f"{source_maps_scens}_{args.num_scens}"

    # Initialize experiment logging directory
    LE = f"data_collection/data/logs/{args.expName}"
    print(LE)
    os.makedirs(LE, exist_ok=True)

    # Setup system resources
    num_cores = multiprocessing.cpu_count()
    print("Current Path:", os.getcwd())

    # Initialize timing and constant paths
    log_time("begin")
    constantMapAndBDFolder = "data_collection/data/benchmark_data/constant_npzs"
    constantMapNpz = f"{constantMapAndBDFolder}/all_maps.npz"

    # Create directory structure for current iteration
    iterFolder = f"{LE}/iter{args.iternum}"
    if not os.path.exists(iterFolder):
        os.makedirs(iterFolder)
    eecbs_outputs_folder = f"{iterFolder}/eecbs_outputs"
    eecbs_path_npzs_folder = f"{iterFolder}/eecbs_npzs"
    processed_folder = f"{iterFolder}/processed"
    model_folder = f"{iterFolder}/models"
    pymodel_outputs_folder = f"{iterFolder}/pymodel_outputs"
    encountered_scens = f"{iterFolder}/encountered_scens"

    # Run EECBS batch processing
    # For iteration 0, use source scenarios
    # For other iterations, use scenarios from previous iteration
    if args.iternum == 0:
        # Initial iteration processing
        command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner",
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                        f"--numAgents={args.numAgents}",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={eecbs_outputs_folder}",
                        f"--num_parallel_runs={args.num_parallel}", f"--iter={args.iternum}",
                        "\"eecbs\"",
                        f"--outputPathNpzFolder={eecbs_path_npzs_folder}",
                        "--firstIter=false",
                        "--cutoffTime=120",
                        f"--suboptimality={args.suboptimality}"])
        print(command)
        subprocess.run(command, shell=True, check=True)
    else:
        # Subsequent iteration processing
        previous_encountered_scens = f"{LE}/iter{args.iternum-1}/encountered_scens"
        command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner",
                    f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={previous_encountered_scens}",
                    f"--numAgents={args.numAgents}",
                    f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                    f"--outputFolder={eecbs_outputs_folder}",
                    f"--num_parallel_runs={args.num_parallel}", f"--iter={args.iternum}",
                    "\"eecbs\"",
                    f"--outputPathNpzFolder={eecbs_path_npzs_folder}",
                    f"--firstIter=false",
                    f"--cutoffTime=120",
                    f"--suboptimality={args.suboptimality}"])
        print(command)
        subprocess.run(command, shell=True, check=True)
    log_time(f"Iter {args.iternum}: Finished eecbs")

    # Clean up EECBS outputs
    command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner",
                    f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                    f"--numAgents={args.numAgents}",
                    f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                    f"--outputFolder={eecbs_outputs_folder}",
                    f"--num_parallel_runs={args.num_parallel}", f"--iter={args.iternum}",
                    "\"clean\" --keepNpys=false"])
    subprocess.run(command, shell=True, check=True)

    # Process data with the dataloader
    command = " ".join(["python", "-m", "gnn.dataloader",
                        f"--mapNpzFile={constantMapNpz}",
                        f"--bdNpzFolder={constantMapAndBDFolder}",
                        f"--pathNpzFolder={eecbs_path_npzs_folder}",
                        f"--processedFolder={processed_folder}",
                        f"--k={args.k}",
                        f"--m={args.m}"])
    if args.extra_layers is not None:
        command += f" --extra_layers={args.extra_layers}"
    if args.bd_pred is not None:
        command += f" --bd_pred={args.bd_pred}"
    print(command)
    subprocess.run(command, shell=True, check=True)

    # Update processed folders list and proceed to training
    processed_folders_list = np.load(LE+"/processed_folders_list.npy")
    processed_folders_list = np.append(processed_folders_list, processed_folder)
    np.save(LE+"/processed_folders_list", processed_folders_list)

    # Generate and execute training script
    log_time(f"Iter {args.iternum}: dataloader")
    generate_sh_script(LE,"run_main", "sbatch_master_process_runner.py", args, chosen_section="train")
    startup_small(LE)
