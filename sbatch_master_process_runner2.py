import os
import subprocess
import argparse
import pdb
import shutil
import multiprocessing
import datetime
import time
import numpy as np
import itertools

from custom_utils.common_helper import str2bool

last_recorded_time = datetime.datetime.now()


def run_maze_generator(args):
    if args.clean:
        try:
            shutil.rmtree(f'{args.maze_data_path}')
        except: pass
    
    maze_cmd = f'''python -m data_collection.maze_generator --data_path={args.maze_data_path} \\
        --maze_config_csv={args.maze_config_csv} \\
        --eecbs_path=./data_collection/eecbs/{args.eecbs_build_release}/eecbs \\
        --temp_bd_path={args.temp_bd_path}/ \\
        --skip_octile_bfs'''

    return maze_cmd


def run_eecbs_batchrunner(args):
    if args.clean:
        try:
            shutil.rmtree(f'{args.exp_path}/iter{args.iternum}/eecbs_npzs')
        except: pass
        try: 
            shutil.rmtree(f'{args.exp_path}/iter{args.iternum}/eecbs_outputs')
        except: pass

    batchrunner_cmd = f'''python -m data_collection.eecbs_batchrunner5 --mapFolder={args.data_path}/maps \\
        --scenFolder={args.data_path}/scens \\
        --numAgents={args.num_agents} \\
        --outputFolder={args.exp_path}/iter{args.iternum}/eecbs_outputs \\
        --num_parallel_runs={args.num_parallel} \\
        "eecbs" \\
        --eecbsPath=./data_collection/eecbs/{args.eecbs_build_release}/eecbs \\
        --outputPathNpzFolder={args.exp_path}/iter{args.iternum}/eecbs_npzs \\
        --cutoffTime=5'''

    return batchrunner_cmd


def run_constants_generator(args):
    if args.clean:
        try: 
            shutil.rmtree(f'{args.temp_bd_path}')
        except: pass
        try: 
            shutil.rmtree(f'{args.data_path}/constant_npzs/')
        except: pass

    constants_cmd = f'''python -m data_collection.constants_generator \\
        --mapFolder={args.data_path}/maps \\
        --scenFolder={args.data_path}/scens \\
        --constantMapAndBDFolder={args.data_path}/constant_npzs \\
        --outputFolder={args.temp_bd_path}/ \\
        --num_parallel_runs={args.num_parallel} \\
        --deleteTextFiles=true \\
        "eecbs" \\
        --eecbsPath=./data_collection/eecbs/{args.eecbs_build_release}/eecbs \\
        --cutoffTime=1'''

    return constants_cmd
    

def run_dataloader(num_multi_inputs, num_multi_outputs, args):
    if args.clean:
        try:
            os.remove(f'{args.exp_path}/iter{args.iternum}/status_data_processed_{num_multi_inputs}_{num_multi_outputs}.csv')
        except: pass
        try: 
            shutil.rmtree(f'{args.exp_path}/iter{args.iternum}/processed_{num_multi_inputs}_{num_multi_outputs}')
        except: pass

    dataloader_cmd = f'''python -m gnn.dataloader --mapNpzFile={args.data_path}/constant_npzs/all_maps.npz \\
        --bdNpzFolder={args.data_path}/constant_npzs \\
        --pathNpzFolder={args.exp_path}/iter{args.iternum}/eecbs_npzs \\
        --processedFolder={args.exp_path}/iter{args.iternum}/processed_{num_multi_inputs}_{num_multi_outputs} \\
        --k={args.k} \\
        --m={args.m} \\
        --num_priority_copies={args.num_priority_copies} {args.bd_pred * '--bd_pred'} \\
        --num_multi_inputs={num_multi_inputs} \\
        --num_multi_outputs={num_multi_outputs}'''
    
    return dataloader_cmd


def run_trainer(num_multi_inputs, num_multi_outputs, args):
    if args.clean:
        try:
            shutil.rmtree(f'{args.exp_path}/iter{args.iternum}/models_{args.model}_{num_multi_inputs}_{num_multi_outputs}{"_p"*args.use_edge_attr}')
        except: pass

    trainer_cmd = f'''python -m gnn.trainer --exp_folder={args.exp_path} --experiment=exp0 --iternum={args.iternum} --num_cores=4 \\
        --processedFolders={args.exp_path}/iter{args.iternum}/processed_{num_multi_inputs}_{num_multi_outputs} \\
        --k={args.k} --m={args.m} --lr={args.lr} \\
        --num_priority_copies={args.num_priority_copies} {args.bd_pred * '--bd_pred'}\\
        --num_multi_inputs={num_multi_inputs} \\
        --num_multi_outputs={num_multi_outputs} \\
        --gnn_name="{args.model}" {args.logging*'--logging'} {args.use_edge_attr*'--use_edge_attr'}'''
    
    return trainer_cmd


def run_simulator(num_multi_inputs, num_multi_outputs, sim_num_agents, args):
    if args.clean:
        try:
            shutil.rmtree(f'{args.exp_path}/tests')
        except: pass

    sim_map = args.sim_scenname.strip().split('-')[0]
    simulator_cmd = f'''python -m gnn.simulator3 --mapNpzFile={args.sim_data_path}/constant_npzs/all_maps.npz \\
        --mapName={sim_map} --scenFile={args.sim_data_path}/scens/{args.sim_scenname}.scen \\
        --agentNum={sim_num_agents} --bdPath={args.sim_data_path}/constant_npzs/ \\
        --k={args.k} --m={args.m} \\
        --outputCSVFile={args.exp_path}/tests/results.csv \\
        --outputPathsFile={args.exp_path}/tests/encountered_scens/paths.npy \\
        --numScensToCreate=10 --outputScenPrefix={args.exp_path}/iter0/encountered_scens/{sim_map}/{args.sim_scenname} \\
        --maxSteps=400 --seed=0 --lacamLookahead=5 --timeLimit=100 {args.bd_pred * '--bd_pred'} \\
        --num_priority_copies=10 \\
        --useGPU=False --modelPath={args.exp_path}/iter0/models_{args.model}_{num_multi_inputs}_{num_multi_outputs}{"_p"*args.use_edge_attr}/max_test_acc.pt \\
        --num_multi_inputs={num_multi_inputs} --num_multi_outputs={num_multi_outputs} --shieldType={args.shield_type}'''
    
    return simulator_cmd


def log_time(exp_path, event_name):
    cur_time = datetime.datetime.now()
    with open(f"{exp_path}/timing.txt", mode='a') as file:
        file.write(f"{event_name} recorded at {cur_time}. \t\t Duration: \t {(cur_time-last_recorded_time).total_seconds()} \n")

def run_command(command):
    # Run the command using subprocess
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print the result of the command
    if result.returncode == 0:
        print(f"Job submitted successfully: {result.stdout}")
    else:
        print(f"Failed to submit job: {result.stderr}")

def generate_sh_script(exp_path, file, conda_env, commands):
    # Open or create the train.sh file in write mode
    os.makedirs(exp_path, exist_ok=True)
    sh_filename = f'{exp_path}/{file}.sh'
    
    if os.path.exists(sh_filename):
        os.remove(sh_filename)
    with open(sh_filename, 'w') as f:
        # Start the script with the command to run the Python script
        f.write("#!/bin/bash\n\n")
        f.write("module load anaconda3/2022.10\n")
        f.write(f"conda activate {conda_env}\n")
        f.write("export MKL_SERVICE_FORCE_INTEL=1\n\n")
        
        for command in commands:
            f.write(f"{command} \n\n")

### Example command for full benchmark
""" 
Generate mazes:
python sbatch_master_process_runner2.py --machine_setting='PSC' --which_setting='Michelle' \
    --exp_dir=EXP_Generate_mazes \
    --maze_data_dir=maze_benchmark_data \
    --maze_config_csv=mazes.csv \
    --clean --which_sections=maze

Small run: 
python sbatch_master_process_runner2.py --machine_setting='PSC' --which_setting='Michelle' \
    --model=ResGatedGraphConv --use_edge_attr \
    --num_multi_inputs_list=0,3 --num_multi_outputs_list=1,2 --bd_pred \
    --clean --which_sections=eecbs,load,train,simulate \
    --sim_scenname='maze4_32_32_1-random-1' --sim_num_agents=10

Small run sections:
python sbatch_master_process_runner2.py --machine_setting='PSC' --which_setting='Michelle' \
    --clean --which_sections=eecbs
python sbatch_master_process_runner2.py --machine_setting='PSC' --which_setting='Michelle' \
    --model=ResGatedGraphConv --use_edge_attr \
    --num_multi_inputs_list=0,3 --num_multi_outputs_list=1,2 --bd_pred \
    --clean --which_sections=load,train
python sbatch_master_process_runner2.py --machine_setting='PSC' --which_setting='Michelle' \
    --model=ResGatedGraphConv --use_edge_attr \
    --num_multi_inputs_list=0,3 --num_multi_outputs_list=1,2 --bd_pred \
    --clean --which_sections=simulate \
    --sim_scenname='maze4_32_32_1-random-1' --sim_num_agents=6

Full run: 
python sbatch_master_process_runner2.py --machine_setting='PSC' --which_setting='Michelle' \
    --data_dir=benchmark_data --exp_dir=EXP_full \
    --model=ResGatedGraphConv --use_edge_attr \
    --num_multi_inputs_list=0,3 --num_multi_outputs_list=1,2 \
    --clean --which_sections=eecbs.load,train,simulate \
    --sim_scenname='maze4_32_32_1-random-1'


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setting and paths
    parser.add_argument('--machine_setting', help="[omega, psc]", required=True, type=str)
    parser.add_argument('--which_setting', help="[Arthur, Rishi, Michelle, PSC]", required=True) # E.g. use --which_setting to determine using 
    parser.add_argument('--data_dir', type=str, default='mini_benchmark_data', help='directory name in data/ that contains maps and scens')
    parser.add_argument('--temp_bd_dir', type=str, default='EXP_Collect_BD', help='directory name in data/logs for constants_generator.py')
    parser.add_argument('--exp_dir', type=str, default='EXP_mini', help='directory name in data/logs for experiment')
    parser.add_argument('--maze_data_dir', type=str, default='maze_benchmark_data', help='directory name in data/ for maze_generator.py')
    parser.add_argument('--maze_config_csv', type=str, default='mazes.csv', help='mazes to generate with maze_generator.py')
    parser.add_argument('--sim_data_dir', type=str, default='maze_benchmark_data', help='directory name in data/ for simulator.py')

    # use default 
    parser.add_argument('--num_parallel', type=int, default=50)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--relu_type', type=str, default="relu")
    extraLayersHelp = "Types of additional layers for training, comma separated. Options are: agent_locations, agent_goal, at_goal_grid"
    parser.add_argument('--extra_layers', help=extraLayersHelp, type=str, default=None)
    parser.add_argument('--shield_type', type=str, default='CS-PIBT')
    parser.add_argument('--iternum', type=int, default=0)
    parser.add_argument('--suboptimality', help="eecbs suboptimality level", type=float, default=2)
    parser.add_argument('--dataset_size', type=int, default=-1)
    # parser.add_argument('--percent_for_succ', help="percent decreased scen creation for success instances in simulation", type=float, required=True)
    # parser.add_argument('--timeLimit', help="time limit for simulation cs-pibt (-1 for no limit)", type=int, required=True)

    # test
    num_agents_help = "Number of agents per scen; [int1,int2,..] or `increment` for all agents up to the max or include .json for pulling from config file, see eecbs_batchrunner3.py "
    parser.add_argument('--num_agents', help=num_agents_help, type=str, default='50,100')
    parser.add_argument('--bd_pred', action="store_true", help="bd_predictions added to NN")
    parser.add_argument('--model', type=str, default='ResGatedGraphConv')
    parser.add_argument('--use_edge_attr', action='store_true')
    parser.add_argument('--num_priority_copies', type=int, default=10)
    parser.add_argument('--num_multi_inputs_list', type=str, help="comma separated numbers of model inputs", default='0')
    parser.add_argument('--num_multi_outputs_list', type=str, help="comma separated numbers of model outputs", default='1')
    parser.add_argument('--sim_scenname', type=str, help="number of agents for simulator.py", default='')
    parser.add_argument('--sim_num_agents', type=str, help="number of agents for simulator.py", default='50')

    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--which_sections', help="[eecbs, load, train, simulate, mazes]", required=True)

    args = parser.parse_args()

    # settings
    if args.machine_setting == 'Omega':
        args.data_path = 'data_collection/data/' + args.data_dir
        args.temp_bd_path = 'data_collection/data/logs/' + args.temp_bd_dir
        args.exp_path = 'data_collection/data/logs/' + args.exp_dir
        args.maze_data_path = 'data_collection/data/' + args.maze_data_dir
        args.maze_config_csv = 'data_collection/data/' + args.maze_config_csv
        args.sim_data_path = 'data_collection/data/' + args.sim_data_dir
        args.eecbs_build_release = 'build_release4'
    elif args.machine_setting == 'PSC':
        project = os.getenv('PROJECT')
        args.data_path = f'{project}/data/' + args.data_dir
        args.temp_bd_path = f'{project}/data/logs/' + args.temp_bd_dir
        args.exp_path = f'{project}/data/logs/' + args.exp_dir
        args.maze_data_path = f'{project}/data/' + args.maze_data_dir
        args.maze_config_csv = f'{project}/data/' + args.maze_config_csv
        args.sim_data_path = f'{project}/data/' + args.sim_data_dir
        args.eecbs_build_release = 'build_release5'
    else:
        raise ValueError(f"Invalid setting: {args.machine_setting}")

    print()
    print("data path:", args.data_path)
    print("exp path:", args.exp_path)

    if args.which_setting == "Arthur":
        conda_env = None # Used in eecbs_batchrunner3 for simulator2.py
    elif args.which_setting == "Rishi":
        conda_env = "pytorchfun"
    elif args.which_setting == "Michelle":
        conda_env = "$PROJECT/.conda/envs/gnn-mapf-dev2"
    elif args.which_setting == "PSC":
        pass
    else:
        raise ValueError(f"Invalid setting: {args.which_setting}")

    # if ".json" in args.numAgents and "map_configs" not in args.numAgents:
    #     args.numAgents = "map_configs/"+args.numAgents 

    if args.data_dir == 'mini_benchmark_data':
        args.num_parallel = 1
    
    # get commands for sh script
    sections = args.which_sections.strip().split(',')
    python_commands = []
    if 'maze' in sections:
        python_commands.append(run_maze_generator(args))
    if 'eecbs' in sections:
        python_commands.append(run_eecbs_batchrunner(args))
        python_commands.append(run_constants_generator(args))

    inputs_outputs = list(itertools.product(args.num_multi_inputs_list.strip().split(','),
                                            args.num_multi_outputs_list.strip().split(',')))
                                        
    if 'load' in sections:
        for num_in, num_out in inputs_outputs:
            python_commands.append(run_dataloader(num_in, num_out, args))
    if 'train' in sections:
        EDGE_ATTR_GNNS = ["ResGatedGraphConv", "GATv2Conv", "TransformerConv", "GENConv"]
        NO_EDGE_ATTR_GNNS = ["SAGEConv"]
        assert(args.model in EDGE_ATTR_GNNS or args.model in NO_EDGE_ATTR_GNNS)
        if args.use_edge_attr: assert(args.model in EDGE_ATTR_GNNS)
        for num_in, num_out in inputs_outputs:
            python_commands.append(run_trainer(num_in, num_out, args))
    if 'simulate' in sections:
        assert(args.sim_scenname)
        for num_in, num_out in inputs_outputs:
            for sim_num_agents in args.sim_num_agents.strip().split(','):
                python_commands.append(run_simulator(num_in, num_out, sim_num_agents, args))

    job_name = f'{args.which_sections}'
    generate_sh_script(args.exp_path, args.which_sections, conda_env, python_commands)

    if args.data_dir == 'mini_benchmark_data':
        command = f'sbatch --job-name {job_name} {args.exp_path}/{args.which_sections}.sh'
    else:
        sbatch_timeout = 16
        command = f'sbatch -p RM-shared -N 1 --ntasks-per-node=10 -t {sbatch_timeout}:00:00 ' + \
        f'--job-name {job_name} {args.exp_path}/{args.which_sections}.sh'

    print('sbatch command:', command, '\n')
    run_command(command.split())
    
    log_time(args.exp_path, "begin")
    