import os
import numpy as np
import subprocess
from multiprocessing import Pool

def generate_sh_script(LE, file, python_file, args):
    # Open or create the train.sh file in write mode
    if os.path.exists(f'{LE}/{file}.sh'):
        os.remove(f'{LE}/{file}.sh')
    with open(f'{LE}/{file}.sh', 'w') as f:
        # Start the script with the command to run the Python script
        f.write("#!/bin/bash\n\n")
        f.write("module load anaconda3\n")
        f.write("conda activate arthur_env\n\n")
        f.write("export MKL_SERVICE_FORCE_INTEL=1\n")
        f.write(f"python -m {python_file} {args}")
        
def run_command(command):
    # Run the command using subprocess
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the result of the command
    if result.returncode == 0:
        print(f"Job submitted successfully: {result.stdout}")
    else:
        print(f"Failed to submit job: {result.stderr}")
        
        
agent_counts = [1,4,16,128]
LE = "benchmarking"
command_list = []
for agent_num in agent_counts:
    args = f" --mapname=all --numAgents=increment --data_folder=data_collection/data/benchmark_original --map_folder=data_collection/data/benchmark_original/maps --scen_folder=data_collection/data/benchmark_original/scens --extra_layers=agent_locations --model_path=data_collection/data/logs/multi_scen_runs/{agent_num}/models/max_test_acc.pt --num_parallel=64 --bd_pred=t --eecbs_cutoff=120 --simulator_cutoff=120 --pymodel_out=benchmarking/{agent_num}_CSFreeze_results --shieldType=CS-Freeze"
    generate_sh_script(LE, f"run_{agent_num}", "benchmarking.benchmark_runner", args)
    command = [
            'sbatch',
            '-p', 'RM-shared',
            '-N', '1',
            '--ntasks-per-node=64',
            '-t', '16:00:00',
            '--job-name', f'arthur_{agent_num}',
            f'./{LE}/run_{agent_num}.sh'
        ]
    command_list.append(command)
    
with Pool() as pool:
    results = pool.map(run_command, command_list)