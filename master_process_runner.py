import os
import subprocess
import argparse
import pdb
import shutil
import multiprocessing
import datetime

from custom_utils.common_helper import str2bool

last_recorded_time = datetime.datetime.now()

def log_time(event_name):
    cur_time = datetime.datetime.now()
    with open(f"./{LE}/timing.txt", mode='a') as file:
        file.write(f"{event_name} recorded at {cur_time}. \t\t Duration: \t {(cur_time-last_recorded_time).total_seconds()} \n")


### Example command for full benchmark
""" 
Small run: python -m master_process_runner 0 t t 100 1000 --num_parallel=50
Big run: python -m master_process_runner 0 f t 100 1000 --num_parallel=50
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expnum", help="experiment number", type=int)
    parser.add_argument('mini_test', type=lambda x: bool(str2bool(x)))
    parser.add_argument('generate_initial', type=lambda x: bool(str2bool(x)))
    parser.add_argument('num_samples', help="number of scens to simulate", type=int)
    parser.add_argument('max_samples', help="max number of scens to simulate", type=int)
    parser.add_argument('--num_parallel', type=int)
    args = parser.parse_args()
    expnum, mini_test, generate_initial, num_samples, max_samples = args.expnum, args.mini_test, args.generate_initial, args.num_samples, args.max_samples
    print(args.expnum)

    iternum = 0
    if mini_test:
        source_maps_scens = "./data_collection/data/mini_benchmark_data"
    else: 
        source_maps_scens = "./data_collection/data/benchmark_data"

    LE = f"data_collection/data/logs/EXP_Small"
    os.makedirs(LE, exist_ok=True)
    
    num_cores = multiprocessing.cpu_count()
    first_iteration = "true"
    print("Current Path:", os.getcwd())

    # command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
    #                     f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
    #                     f"--num_parallel={args.num_parallel}", "--cutoffTime=20"])
    # print(command)
    # print(command)
    # pdb.set_trace()
    
    log_time("begin")
    constantMapAndBDFolder = "data_collection/data/benchmark_data/constant_npzs"
    constantMapNpz = f"{constantMapAndBDFolder}/all_maps.npz"
    
    # if generate_initial:
    #     command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
    #                     f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
    #                     f"--constantMapAndBDFolder={constantMapAndBDFolder}",
    #                     f"--outputFolder={LE}/iter0/eecbs_outputs", 
    #                     f"--num_parallel_runs={args.num_parallel}",
    #                     "\"eecbs\"",
    #                     f"--outputPathNpzFolder={LE}/iter0/eecbs_npzs",
    #                     f"--firstIter={first_iteration}",
    #                     f"--cutoffTime=20",
    #                     f"--suboptimality=2"])
    #     ### Example command for minibenchmark
    #     # python ./data_collection/eecbs_batchrunner2.py --mapFolder=./data_collection/data/mini_benchmark_data/maps 
    #     #           --scenFolder=./data_collection/data/mini_benchmark_data/scens --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ 
    #     #           --expnum=0 --firstIter=true --iter=0 --num_parallel=50 --cutoffTime=20
    #     ### Example command for benchmark
    #     # python ./data_collection/eecbs_batchrunner2.py --mapFolder=./data_collection/data/benchmark_data/maps 
    #     #           --scenFolder=./data_collection/data/benchmark_data/scens --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ 
    #     #           --expnum=0 --firstIter=true --iter=0 --num_parallel=50 --cutoffTime=20
    #     print(command)
    #     subprocess.run(command, shell=True, check=True)
    # else:
    #     assert(os.path.exists(f"{LE}/iter0/eecbs_npzs/warehouse_10_20_10_2_2_paths.npz"))
    
    # log_time("initial generation")
    # print("Done with first eecbs run")
    # # pdb.set_trace()

    processed_folders_list = []
    for iternum in range(30):
        iterFolder = f"{LE}/iter{iternum}"
        if not os.path.exists(iterFolder):
            os.makedirs(iterFolder)
        eecbs_outputs_folder = f"{iterFolder}/eecbs_outputs"
        eecbs_path_npzs_folder = f"{iterFolder}/eecbs_npzs"
        processed_folder = f"{iterFolder}/processed"
        model_folder = f"{iterFolder}/models"
        pymodel_outputs_folder = f"{iterFolder}/pymodel_outputs"
        encountered_scens = f"{iterFolder}/encountered_scens"

        ### Run EECBS labeller
        if iternum == 0: # Special logic in the first iteration
            if args.generate_initial:
                # Run initial data collection
                command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
                                f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                                f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                                f"--outputFolder={eecbs_outputs_folder}", 
                                f"--num_parallel_runs={args.num_parallel}",
                                "\"eecbs\"",
                                f"--outputPathNpzFolder={eecbs_path_npzs_folder}",
                                "--firstIter=true",
                                "--cutoffTime=20",
                                "--suboptimality=2"])
                print(command)
                subprocess.run(command, shell=True, check=True)
            else:
                # Make sure that the initial data was created
                assert(os.path.exists(f"{LE}/iter0/eecbs_npzs/warehouse_10_20_10_2_2_paths.npz"))
        else:
            ### Run eecbs labeler on encountered scens
            previous_encountered_scens = f"{LE}/iter{iternum-1}/encountered_scens"
            command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={previous_encountered_scens}",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={eecbs_outputs_folder}", 
                        f"--num_parallel_runs={args.num_parallel}",
                        "\"eecbs\"",
                        f"--outputPathNpzFolder={eecbs_path_npzs_folder}",
                        f"--firstIter=false",
                        f"--cutoffTime=20",
                        f"--suboptimality=2"])
            print(command)
            subprocess.run(command, shell=True, check=True)
        log_time(f"Iter {iternum}: Finished eecbs")

        ### Process the data, i.e. create pt files from path npzs
        command = " ".join(["python", "-m", "gnn.dataloader", 
                            f"--mapNpzFile={constantMapNpz}", 
                            f"--bdNpzFolder={constantMapAndBDFolder}",
                            f"--pathNpzFolder={eecbs_path_npzs_folder}",
                            f"--processedFolder={processed_folder}"])
        print(command)
        subprocess.run(command, shell=True, check=True)
        processed_folders_list.append(processed_folder)
        log_time(f"Iter {iternum}: dataloader")

        ### Train the model
        command = " ".join(["python", "-m", "gnn.trainer", f"--exp_folder={LE}", f"--experiment=exp{expnum}", 
                            f"--iternum={iternum}", f"--num_cores={num_cores}", 
                            f"--processedFolders={','.join(processed_folders_list)}"])
        print(command)
        subprocess.run(command, shell=True, check=True)
        log_time(f"Iter {iternum}: trainer")
        # pdb.set_trace()

        ### Run best model on simulator on scens to create new scenes
        command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={pymodel_outputs_folder}", 
                        f"--num_parallel_runs={min(20, args.num_parallel)}",
                        "\"pymodel\"",
                        f"--modelPath={iterFolder}/models/max_test_acc.pt",
                        "--useGPU=False",
                        "--k=4",
                        "--m=5",
                        "--maxSteps=200",
                        "--numScensToCreate=10",])
        print(command)
        subprocess.run(command, shell=True, check=True)
        log_time(f"Iter {iternum}: simulator")


        ### Create a folder with scen files for the next iteration
        os.makedirs(encountered_scens, exist_ok=True)
        for mapfolder in os.listdir(pymodel_outputs_folder):
            folder_path = f"{pymodel_outputs_folder}/{mapfolder}"
            if os.path.isdir(folder_path):
                assert(os.path.exists(f"{folder_path}/paths"))
                for file in os.listdir(f"{folder_path}/paths"):
                    if file.endswith(".scen"):
                        shutil.copy(f"{folder_path}/paths/{file}", f"{encountered_scens}/{file}")
        
        # pdb.set_trace()
        # feed failures into eecbs
        # iternum+=1
        # command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
        #                 f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
        #                 f"--constantMapAndBDFolder={constantMapAndBDFolder}",
        #                 f"--outputFolder={nextIterFolder}/eecbs_outputs", 
        #                 f"--num_parallel_runs={args.num_parallel}",
        #                 "\"eecbs\"",
        #                 f"--outputPathNpzFolder={nextIterFolder}/eecbs_npzs",
        #                 f"--firstIter={first_iteration}",
        #                 f"--cutoffTime=20",
        #                 f"--suboptimality=2"])
        # Note: scen's should be taken from previous iteration but npz should be saved this next iteration number (since npzs have an extra)
        # subprocess.run(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}"]) # TODO figure out where eecbs is outputting files
        # command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
        #                 f"--num_parallel={args.num_parallel}", "--cutoffTime=20"])
        # subprocess.run(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", 
        #                 f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
        #                 f"--num_parallel_runs={args.num_parallel}", "--cutoffTime=20"]) # TODO figure out where eecbs is outputting files
        # log_time(f"Iter {iternum}: eecbs")