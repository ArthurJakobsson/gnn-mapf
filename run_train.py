# Standard library imports
import os
import subprocess
import argparse
import pdb
import shutil
import multiprocessing
import datetime
import time
import numpy as np
import sys

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
    if args.which_setting == "Arthur":
        conda_env = None # Used in eecbs_batchrunner3 for simulator2.py
    elif args.which_setting == "Rishi":
        conda_env = "pytorchfun"
    elif args.which_setting == "PSC":
        pass
    else:
        raise ValueError(f"Invalid setting: {args.which_setting}")

    if ".json" in args.numAgents and "map_configs" not in args.numAgents:
        args.numAgents = "map_configs/"+args.numAgents

    if args.mini_test:
        # source_maps_scens = "./data_collection/data/mini_benchmark_data"
        source_maps_scens = f"./data_collection/data/{args.data_folder}"
    else:
        source_maps_scens = "./data_collection/data/benchmark_data"


    source_maps_scens = f"{source_maps_scens}_{args.num_scens}"
    LE = f"data_collection/data/logs/{args.expName}"
    os.makedirs(LE, exist_ok=True)

    num_cores = multiprocessing.cpu_count()
    first_iteration = "true"
    print("Current Path:", os.getcwd())


    log_time("begin")
    constantMapAndBDFolder = "data_collection/data/benchmark_data/constant_npzs"
    constantMapNpz = f"{constantMapAndBDFolder}/all_maps.npz"

    iterFolder = f"{LE}/iter{args.iternum}"
    if not os.path.exists(iterFolder):
        os.makedirs(iterFolder)
    eecbs_outputs_folder = f"{iterFolder}/eecbs_outputs"
    eecbs_path_npzs_folder = f"{iterFolder}/eecbs_npzs"
    processed_folder = f"{iterFolder}/processed"
    model_folder = f"{iterFolder}/models_{args.dataset_size}"
    pymodel_outputs_folder = f"{iterFolder}/pymodel_outputs"
    encountered_scens = f"{iterFolder}/encountered_scens"
    processed_folders_list = np.load(LE+"/processed_folders_list.npy")
    # processed_folders_list = processed_folders_list[-3:] # Only keep the last 1 iterations

    command = " ".join(["python", "-m", "gnn.trainer", f"--exp_folder={LE}", f"--experiment=exp{args.expnum}",
                            f"--iternum={args.iternum}", f"--num_cores={num_cores}",
                            f"--processedFolders={','.join(processed_folders_list)}",
                            f"--k={args.k}", f"--m={args.m}", f"--lr={args.lr}", f"--relu_type={args.relu_type}",
                            f"--dataset_size={args.dataset_size}"])
    if args.extra_layers is not None:
        command += f" --extra_layers={args.extra_layers}"
    if args.bd_pred is not None:
        command += f" --bd_pred={args.bd_pred}"
    print(command)
    subprocess.run(command, shell=True, check=True)
    log_time(f"Iter {args.iternum}: trainer")

    quit()
    generate_sh_script(LE, "run_main", "sbatch_master_process_runner.py", args, chosen_section="simulate")
    startup_small(LE)
