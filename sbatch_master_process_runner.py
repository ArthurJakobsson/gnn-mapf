import os
import subprocess
import argparse
import pdb
import shutil
import multiprocessing
import datetime
import time
import numpy as np

from custom_utils.common_helper import str2bool
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

def run_command(command):
    """
    Executes a shell command and handles the output
    Args:
        command: Command to execute as list of strings
    """
    print(command)
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Job submitted successfully: {result.stdout}")
    else:
        print(f"Failed to submit job: {result.stderr}")

def startup_small(LE):
    """
    Submits a small job to the RM-shared partition
    Args:
        LE: Log directory path
    """
    command = [
        'sbatch',
        '-p', 'RM-shared',
        '-N', '1',
        '--ntasks-per-node=2',
        '-t', '0:30:00',
        '--job-name', 'arthur_main',
        f'./{LE}/run_main.sh'
    ]
    run_command(command)



def generate_sh_script(LE, file, python_file, args, chosen_section=None):
    """
    Generates a shell script for job submission
    Args:
        LE: Log directory path
        file: Output shell script name
        python_file: Python script to run
        args: Command line arguments
        chosen_section: Section to run (optional)
    """
    if os.path.exists(f'{LE}/{file}.sh'):
        os.remove(f'{LE}/{file}.sh')
    with open(f'{LE}/{file}.sh', 'w') as f:
        # Write shell script header
        f.write("#!/bin/bash\n\n")
        f.write("module load anaconda3\n")
        f.write("conda activate arthur_env\n\n")
        f.write("export MKL_SERVICE_FORCE_INTEL=1\n")
        f.write(f"python {python_file} \\\n")

        # Add command line arguments
        for key, value in vars(args).items():
            if value is None:
                continue
            if "which_section" in key and chosen_section is not None:
                f.write(f" --{key}={chosen_section}\\\n")
            elif isinstance(value, bool):
                if value:
                    f.write(f" --{key}=t\\\n")
                else:
                    f.write(f" --{key}=f\\\n")
            else:
                f.write(f" --{key}={value} \\\n")

        f.seek(f.tell() - 3, 0)
        f.truncate()

### Example command for full benchmark
"""
Small run: python master_process_runner.py 0 t --expName=EXP_400_agents  --data_folder=den312_benchmark --num_parallel=50 --k=4 --m=7 --lr=0.001 --relu_type=leaky_relu --numAgents=100,200,300,400 --which_setting=Arthur --extra_layers=agent_locations --bd_pred=t
Big run: python -m master_process_runner 0 f t 100 1000 --num_parallel=50
Old big run: python -m master_process_runner 0 f f 100 1000 --num_parallel=50
Small run: python -m master_process_runner 0 t --numScensToCreate=10 --num_parallel=10 --expName=EXP_den312d_test6 \
    --numAgents=100 --which_setting=Rishi
"""
if __name__ == "__main__":
    """
    Master process runner for managing multi-stage machine learning experiments on a SLURM cluster.

    This script coordinates the execution of a machine learning pipeline across multiple stages:
    1. Begin: Initializes experiment by creating a processed_folders tracking file
    2. Setup: Prepares training data and environment (run_setup.py)
    3. Train: Executes model training on GPU (run_train.py)
    4. Simulate: Runs simulations to evaluate the trained model (run_simulator.py)

    The script handles SLURM job submission for each stage with appropriate resource allocation:
    - Setup/Simulate: Uses RM-shared partition with CPU resources
    - Training: Uses GPU-shared partition with V100 GPU

    System is designed for PSC computing cluster.

    Each stage is configured through command line arguments that control experiment parameters,
    dataset configuration, model architecture, and training settings.
    """

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

    if args.iternum>10:
        quit()
    if args.mini_test:
        # source_maps_scens = "./data_collection/data/mini_benchmark_data"
        source_maps_scens = f"./data_collection/data/{args.data_folder}"
    else:
        source_maps_scens = "./data_collection/data/benchmark_data"

    if args.num_scens!=0:
        # for each map, save only the first {args.num_scens} scen files
        scen_dir = f"{source_maps_scens}_{args.num_scens}"
        if not os.path.isdir(scen_dir):
            os.makedirs(scen_dir)
            # copy over the maps
            shutil.copytree(f"{source_maps_scens}/maps", f"{scen_dir}/maps", dirs_exist_ok=True)
            # copy over only necessary scens
            os.makedirs(f"{scen_dir}/scens")
            for i in range(26,args.num_scens+26):
                for scen_path in os.listdir(f"{source_maps_scens}/scens"):
                    if scen_path.endswith(f"-{i}.scen"):
                        shutil.copyfile(os.path.join(source_maps_scens, 'scens', scen_path), os.path.join(scen_dir, 'scens', scen_path))
            # reset the source_maps_scens folder
        source_maps_scens = scen_dir


    print(source_maps_scens)
    LE = f"data_collection/data/logs/{args.expName}"
    os.makedirs(LE, exist_ok=True)

    num_cores = multiprocessing.cpu_count()
    first_iteration = "true"
    print("Current Path:", os.getcwd())


    log_time("begin")
    constantMapAndBDFolder = "data_collection/data/benchmark_data/constant_npzs"
    constantMapNpz = f"{constantMapAndBDFolder}/all_maps.npz"

    def call_setup():
        # call sbatch for run_setup
        generate_sh_script(LE,f"setup{args.dataset_size}", "run_setup.py", args)
        command = [
            'sbatch',
            '-p', 'RM-shared',
            '-N', '1',
            '--ntasks-per-node=64',
            '-t', '16:00:00',
            '--job-name', 'arthur_setup',
            f'./{LE}/setup{args.dataset_size}.sh'
        ]
        run_command(command)

    def call_train():
        # call sbatch for run_train
        generate_sh_script(LE,f"train{args.dataset_size}", "run_train.py", args)
        command = [
            'sbatch',
            '-p', 'GPU-shared',
            '--gres=gpu:v100-32:1',
            '-t', '24:00:00',
            '--job-name', 'arthur_train',
            f'./{LE}/train{args.dataset_size}.sh'
        ]
        run_command(command)

    def call_simulate():
        # call sbatch for simulation
        generate_sh_script(LE,f"simulate{args.dataset_size}", "run_simulator.py", args)
        command = [
            'sbatch',
            '-p', 'RM-shared',
            '-N', '1',
            '--ntasks-per-node=64',
            '-t', '16:00:00',
            '--job-name', 'arthur_simulate',
            f'./{LE}/simulate{args.dataset_size}.sh'
        ]
        run_command(command)


    # Main execution logic:
    # 1. Begin: Generate initial processed_folders_list.npy file
    # 2. Setup: Run run_setup.py via bash script
    # 3. Train: Run run_train.py via bash script
    # 4. Simulate: Run run_simulator.py via bash script
    if args.which_section == "begin":
        processed_folders_list = np.array([])
        np.save(LE+"/processed_folders_list", processed_folders_list)
        call_setup()
    elif args.which_section == "setup":
        call_setup()
    elif args.which_section == "train":
        call_train()
    elif args.which_section == "simulate":
        call_simulate()
    else:
        raise ValueError(f"Invalid section: {args.which_section}")
