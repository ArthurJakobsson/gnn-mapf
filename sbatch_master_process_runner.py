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

last_recorded_time = datetime.datetime.now()

def log_time(event_name):
    cur_time = datetime.datetime.now()
    with open(f"./{LE}/timing.txt", mode='a') as file:
        file.write(f"{event_name} recorded at {cur_time}. \t\t Duration: \t {(cur_time-last_recorded_time).total_seconds()} \n")

def run_command(command):
    # Run the command using subprocess
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the result of the command
    if result.returncode == 0:
        print(f"Job submitted successfully: {result.stdout}")
    else:
        print(f"Failed to submit job: {result.stderr}")

def startup_small(LE):
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
    # Open or create the train.sh file in write mode
    if os.path.exists(f'{LE}/{file}.sh'):
        os.remove(f'{LE}/{file}.sh')
    with open(f'{LE}/{file}.sh', 'w') as f:
        # Start the script with the command to run the Python script
        f.write("#!/bin/bash\n\n")
        f.write("module load anaconda3\n")
        f.write("conda activate arthur_env\n\n")
        f.write("export MKL_SERVICE_FORCE_INTEL=1\n")
        f.write(f"python {python_file} \\\n")

        # Iterate over the parsed arguments in args
        for key, value in vars(args).items():
            if value is None:
                continue
            if "which_section" in key and chosen_section is not None:
                # pass # don't modify section in main script
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
    parser.add_argument('--timeLimit', help="time limit for simulation cs-pibt (-1 for no limit)", type=int, required=True)
    parser.add_argument('--num_scens', help="number scens to include, for each map, in the train set", type=int, required=True)

    args = parser.parse_args()
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
    
    # for each map, save only the first {args.num_scens} scen files
    scen_dir = f"{source_maps_scens}_{args.num_scens}"
    if not os.path.isdir(scen_dir):
        os.makedirs(scen_dir)
        # copy over the maps
        shutil.copytree(f"{source_maps_scens}/maps", f"{scen_dir}/maps", dirs_exist_ok=True)
        # copy over only necessary scens
        os.makedirs(f"{scen_dir}/scens")
        for i in range(1,args.num_scens+1):
            for scen_path in os.listdir(f"{source_maps_scens}/scens"):
                if scen_path.endswith(f"{i}.scen"):
                    shutil.copyfile(os.path.join(source_maps_scens, 'scens', scen_path), os.path.join(scen_dir, 'scens', scen_path))
        # reset the source_maps_scens folder
    source_maps_scens = scen_dir

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
        generate_sh_script(LE,"setup", "run_setup.py", args)
        command = [
            'sbatch',
            '-p', 'RM-shared',
            '-N', '1',
            '--ntasks-per-node=64',
            '-t', '16:00:00',
            '--job-name', 'arthur_setup',
            f'./{LE}/setup.sh'
        ]
        run_command(command)
    
    def call_train():
        # call sbatch for run_train
        generate_sh_script(LE,"train", "run_train.py", args)
        command = [
            'sbatch',
            '-p', 'GPU-shared',
            '--gres=gpu:v100-32:1',
            '-t', '24:00:00',
            '--job-name', 'arthur_train',
            f'./{LE}/train.sh'
        ]
        run_command(command)
    
    def call_simulate():
        # call sbatch for simulation
        generate_sh_script(LE,"simulate", "run_simulator.py", args)
        command = [
            'sbatch',
            '-p', 'RM-shared',
            '-N', '1',
            '--ntasks-per-node=64',
            '-t', '16:00:00',
            '--job-name', 'arthur_simulate',
            f'./{LE}/simulate.sh'
        ]
        run_command(command)


    
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
