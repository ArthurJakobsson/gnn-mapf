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
from sbatch_master_process_runner import generate_sh_script, startup_small

from custom_utils.common_helper import str2bool

last_recorded_time = datetime.datetime.now()

def log_time(event_name):
    cur_time = datetime.datetime.now()
    with open(f"./{LE}/timing.txt", mode='a') as file:
        file.write(f"{event_name} recorded at {cur_time}. \t\t Duration: \t {(cur_time-last_recorded_time).total_seconds()} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expnum", help="experiment number", type=int)
    parser.add_argument('--mini_test', type=lambda x: bool(str2bool(x)))
    # parser.add_argument('generate_initial', help="NOTE: We should NOT need to do this given constant_npzs/ folder", type=lambda x: bool(str2bool(x)))
    parser.add_argument('--numScensToCreate', type=int, help="number of scens to create per pymodel, see simulator2.py", default=20)
    parser.add_argument('--num_parallel', type=int)
    parser.add_argument('--data_folder', type=str, help="name of folder with data")
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--relu_type', type=str, default="relu")
    parser.add_argument('--expName', help="Name of the experiment, e.g. Test5", required=True)
    numAgentsHelp = "Number of agents per scen; [int1,int2,..] or `increment` for all agents up to the max or include .json for pulling from config file, see eecbs_batchrunner3.py "
    parser.add_argument('--numAgents', help=numAgentsHelp, type=str, required=True)
    extraLayersHelp = "Types of additional layers for training, comma separated. Options are: agent_locations, agent_goal, at_goal_grid"
    parser.add_argument('--extra_layers', help=extraLayersHelp, type=str, default=None)
    parser.add_argument('--bd_pred', type=str, default=None, help="bd_predictions added to NN, type anything if adding")
    parser.add_argument('--which_setting', help="[Arthur, Rishi, PSC]", required=True) # E.g. use --which_setting to determine using conda env or different aspects
    parser.add_argument('--percent_for_succ', help="percent decreased scen creation for success instances in simulation", type=float, required=True)
    parser.add_argument('--which_section', help="[begin, setup, train, simulate]", required=True)
    parser.add_argument('--iternum', type=int)

    args = parser.parse_args()
    if args.which_setting == "Arthur":
        conda_env = None # Used in eecbs_batchrunner3 for simulator2.py
    elif args.which_setting == "Rishi":
        conda_env = "pytorchfun"
    elif args.which_setting == "PSC":
        pass
    else:
        raise ValueError(f"Invalid setting: {args.which_setting}")

    if ".json" in args.numAgents:
        args.numAgents = "map_configs/"+args.numAgents     

    if args.mini_test:
        # source_maps_scens = "./data_collection/data/mini_benchmark_data"
        source_maps_scens = f"./data_collection/data/{args.data_folder}"
    else: 
        source_maps_scens = "./data_collection/data/benchmark_data"

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
    # processed_folders_list = processed_folders_list[-3:] # Only keep the last 1 iterations
    
    command = " ".join(["python", "-m", "gnn.trainer", f"--exp_folder={LE}", f"--experiment=exp{args.expnum}", 
                            f"--iternum={args.iternum}", f"--num_cores={num_cores}", 
                            f"--processedFolders={','.join(processed_folders_list)}",
                            f"--k={args.k}", f"--m={args.m}", f"--lr={args.lr}", f"--relu_type={args.relu_type}"])
    if args.extra_layers is not None:
        command += f" --extra_layers={args.extra_layers}"
    if args.bd_pred is not None:
        command += f" --bd_pred={args.bd_pred}"
    print(command)
    subprocess.run(command, shell=True, check=True)
    log_time(f"Iter {args.iternum}: trainer")
    
    generate_sh_script(LE, "run_main", "sbatch_master_process_runner.py", args, chosen_section="simulate")
    startup_small(LE)