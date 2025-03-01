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
        conda_env = None # Used in eecbs_batchrunner for simulator2.py
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
    model_folder = f"{iterFolder}/models"
    pymodel_outputs_folder = f"{iterFolder}/pymodel_outputs"
    encountered_scens = f"{iterFolder}/encountered_scens"
    processed_folders_list = np.load(LE+"/processed_folders_list.npy")


    ### Run best model on simulator on scens to create new scenes
    command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner",
                    f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                    f"--numAgents={args.numAgents}",
                    f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                    f"--outputFolder={pymodel_outputs_folder}",
                    f"--num_parallel_runs={min(20, args.num_parallel)}", f"--iter={args.iternum}",
                    "\"pymodel\"",
                    f"--modelPath={iterFolder}/models/max_test_acc.pt",
                    "--useGPU=False",
                    "--k=4",
                    "--m=5",
                    "--maxSteps=3x",
                    "--shieldType=CS-PIBT",
                    f"--timeLimit={args.timeLimit}",
                    # "--lacamLookahead=50",
                    f"--numScensToCreate={args.numScensToCreate}",
                    f"--percentSuccessGenerationReduction={args.percent_for_succ}"])
    if args.extra_layers is not None:
        command += f" --extra_layers={args.extra_layers}"
    if args.bd_pred is not None:
        command += f" --bd_pred={args.bd_pred}"
    if conda_env is not None:
        command += f" --condaEnv={conda_env}"
    print(command)
    subprocess.run(command, shell=True, check=True)
    log_time(f"Iter {args.iternum}: simulator")


    ### Create a folder with scen files for the next iteration
    # Move scens from pymodel_outputs_folder/[MAPNAME] to encountered_scens/
    os.makedirs(encountered_scens, exist_ok=True)
    for mapfolder in os.listdir(pymodel_outputs_folder):
        folder_path = f"{pymodel_outputs_folder}/{mapfolder}"
        if os.path.isdir(folder_path):
            assert(os.path.exists(f"{folder_path}/paths"))
            for file in os.listdir(f"{folder_path}/paths"):
                if file.endswith(".scen"):
                    shutil.copy(f"{folder_path}/paths/{file}", f"{encountered_scens}/{file}")

    ### Clean up the pymodel_outputs_folder
    command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner",
                    f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                    f"--numAgents={args.numAgents}",
                    f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                    f"--outputFolder={pymodel_outputs_folder}",
                    f"--num_parallel_runs={args.num_parallel}", f"--iter={args.iternum}",
                    "\"clean\" --keepNpys=true"])
    subprocess.run(command, shell=True, check=True)

    args.iternum += 1
    generate_sh_script(LE, "run_main", "sbatch_master_process_runner.py", args, chosen_section="setup")
    startup_small(LE)
