import os
import subprocess
import argparse
import pdb
import shutil
import multiprocessing
import datetime
import time
from custom_utils.argparser_main import parse_arguments

last_recorded_time = datetime.datetime.now()

def log_time(event_name):
    cur_time = datetime.datetime.now()
    with open(f"./{LE}/timing.txt", mode='a') as file:
        file.write(f"{event_name} recorded at {cur_time}. \t\t Duration: \t {(cur_time-last_recorded_time).total_seconds()} \n")


### Example command for full benchmark
""" 
Small run: python master_process_runner.py 0 t --expName=EXP_400_agents  --data_folder=den312_benchmark --num_parallel=50 --k=4 --m=7 --lr=0.001 --relu_type=leaky_relu --numAgents=100,200,300,400 --which_setting=Arthur --extra_layers=agent_locations --bd_pred=t --percent_for_succ=0.5
Big run: python -m master_process_runner 0 f t 100 1000 --num_parallel=50
Old big run: python -m master_process_runner 0 f f 100 1000 --num_parallel=50
Small run: python -m master_process_runner 0 t --numScensToCreate=20 --num_parallel=30 --expName=EXP_den312d_lacam2 \
    --numAgents=100,200,400,600,800 --which_setting=Rishi
"""
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
        "timeLimit",
        "suboptimality",
        "dataset_size"
    ]

    args = parse_arguments(flags)
    if args.which_setting == "Arthur":
        conda_env = None # Used in eecbs_batchrunner for simulator.py
    elif args.which_setting == "Rishi":
        conda_env = "pytorchfun"
    elif args.which_setting == "PSC":
        pass
    else:
        raise ValueError(f"Invalid setting: {args.which_setting}")

    if ".json" in args.numAgents:
        args.numAgents = "map_configs/"+args.numAgents
    
    iternum = 0
    if args.mini_test:
        # source_maps_scens = "./data_collection/data/mini_benchmark_data"
        source_maps_scens = f"./data_collection/data/{args.data_folder}"
    else: 
        source_maps_scens = "./data_collection/data/benchmark_data"

    LE = f"data_collection/data/logs/{args.expName}"
    os.makedirs(LE, exist_ok=True)
    
    # num_cores = multiprocessing.cpu_count()
    num_cores=args.num_parallel
    first_iteration = "true"
    print("Current Path:", os.getcwd())

    
    log_time("begin")
    constantMapAndBDFolder = "data_collection/data/benchmark_data/constant_npzs" #TODO change this to make new maps
    constantMapNpz = f"{constantMapAndBDFolder}/all_maps.npz"
    

    processed_folders_list = []
    for iternum in range(10):
        iterFolder = f"{LE}/iter{iternum}"
        if not os.path.exists(iterFolder):
            os.makedirs(iterFolder)
        eecbs_outputs_folder = f"{iterFolder}/eecbs_outputs"
        eecbs_path_npzs_folder = f"{iterFolder}/eecbs_npzs"
        processed_folder = f"{iterFolder}/processed"
        model_folder = f"{iterFolder}/models"
        pymodel_outputs_folder = f"{iterFolder}/pymodel_outputs"
        encountered_scens = f"{iterFolder}/encountered_scens"

        # Run EECBS labeller
        if iternum == 0: # Special logic in the first iteration
            # if args.generate_initial:
            # Run initial data collection
            command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner", 
                            f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                            f"--numAgents={args.numAgents}",
                            f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                            f"--outputFolder={eecbs_outputs_folder}", 
                            f"--num_parallel_runs={args.num_parallel}", f"--iter={iternum}",
                            "\"eecbs\"",
                            f"--outputPathNpzFolder={eecbs_path_npzs_folder}",
                            "--firstIter=false", # Note we should not need to create bds anymore, which is what this is used for
                            "--cutoffTime=120",
                            f"--suboptimality={args.suboptimality}"])
            print(command)
            subprocess.run(command, shell=True, check=True)
            # else:
            #     # Make sure that the initial data was created
            #     assert(os.path.exists(f"{constantMapAndBDFolder}/warehouse_10_20_10_2_2_paths.npz"))
            #     # assert(os.path.exists(f"{LE}/iter0/eecbs_npzs/warehouse_10_20_10_2_2_paths.npz"))
        else:
            ### Run eecbs labeler on encountered scens
            previous_encountered_scens = f"{LE}/iter{iternum-1}/encountered_scens"
            command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner", 
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={previous_encountered_scens}",
                        f"--numAgents={args.numAgents}",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={eecbs_outputs_folder}", 
                        f"--num_parallel_runs={args.num_parallel}", f"--iter={iternum}",
                        "\"eecbs\"",
                        f"--outputPathNpzFolder={eecbs_path_npzs_folder}",
                        f"--firstIter=false",
                        f"--cutoffTime=120",
                        f"--suboptimality={args.suboptimality}"])
            print(command)
            subprocess.run(command, shell=True, check=True)
        log_time(f"Iter {iternum}: Finished eecbs")
        
        
        ### Clean up the eecbs_outputs_folder
        command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner", 
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                        f"--numAgents={args.numAgents}",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={eecbs_outputs_folder}", 
                        f"--num_parallel_runs={args.num_parallel}", f"--iter={iternum}",
                        "\"clean\" --keepNpys=false"])
        subprocess.run(command, shell=True, check=True)
        
        ### Process the data, i.e. create pt files from path npzs
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
        processed_folders_list.append(processed_folder)
        log_time(f"Iter {iternum}: dataloader")

        # processed_folders_list = processed_folders_list[-3:] # Only keep the last 1 iterations

        ### Train the model
        command = " ".join(["python", "-m", "gnn.trainer", f"--exp_folder={LE}", f"--experiment=exp{args.expnum}", 
                            f"--iternum={iternum}", f"--num_cores={num_cores}", 
                            f"--processedFolders={','.join(processed_folders_list)}",
                            f"--k={args.k}", f"--m={args.m}", f"--lr={args.lr}", f"--relu_type={args.relu_type}", 
                            f"--dataset_size={args.dataset_size}"])
        if args.extra_layers is not None:
            command += f" --extra_layers={args.extra_layers}"
        if args.bd_pred is not None:
            command += f" --bd_pred={args.bd_pred}"
        print(command)
        subprocess.run(command, shell=True, check=True)
        log_time(f"Iter {iternum}: trainer")
        # pdb.set_trace()

        ### Run best model on simulator on scens to create new scenes
        command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner", 
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                        f"--numAgents={args.numAgents}",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={pymodel_outputs_folder}", 
                        f"--num_parallel_runs={min(20, args.num_parallel)}", f"--iter={iternum}",
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
        log_time(f"Iter {iternum}: simulator")


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
                        f"--num_parallel_runs={args.num_parallel}", f"--iter={iternum}",
                        "\"clean\" --keepNpys=true"])
        subprocess.run(command, shell=True, check=True)
