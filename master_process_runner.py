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
# python -m master_process_runner 0 f t 100 1000 --num_parallel=50
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

    LE = f"data_collection/data/logs/EXP_Test3"
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
    
    if generate_initial:
        command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={LE}/iter0/eecbs_outputs", 
                        f"--num_parallel_runs={args.num_parallel}",
                        "\"eecbs\"",
                        f"--outputPathNpzFolder={LE}/iter0/eecbs_npzs",
                        f"--firstIter={first_iteration}",
                        f"--cutoffTime=20",
                        f"--suboptimality=2"])
        ### Example command for minibenchmark
        # python ./data_collection/eecbs_batchrunner2.py --mapFolder=./data_collection/data/mini_benchmark_data/maps 
        #           --scenFolder=./data_collection/data/mini_benchmark_data/scens --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ 
        #           --expnum=0 --firstIter=true --iter=0 --num_parallel=50 --cutoffTime=20
        ### Example command for benchmark
        # python ./data_collection/eecbs_batchrunner2.py --mapFolder=./data_collection/data/benchmark_data/maps 
        #           --scenFolder=./data_collection/data/benchmark_data/scens --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ 
        #           --expnum=0 --firstIter=true --iter=0 --num_parallel=50 --cutoffTime=20
        print(command)
        subprocess.run(command, shell=True, check=True)
    else:
        assert(os.path.exists(f"{LE}/iter0/eecbs_npzs/warehouse_10_20_10_2_2_paths.npz"))
    
    log_time("initial generation")
    print("Done with first eecbs run")
    # pdb.set_trace()

    while True:        
        # if not os.path.exists(f"./{LE}/iter{iternum}"):
        #     os.makedirs(f"./{LE}/iter{iternum}")
        iterFolder = f"{LE}/iter{iternum}"
        nextIterFolder = f"{LE}/iter{iternum+1}"
        if not os.path.exists(iterFolder):
            os.makedirs(iterFolder)

        # train the naive model
        # python -m gnn.trainer --exp_folder=data_collection/data/logs/EXP_Test --experiment=exp0 --iternum=0 --num_cores=4 --generate_initial=True
        command = " ".join(["python", "-m", "gnn.trainer", f"--exp_folder={LE}", f"--experiment=exp{expnum}", 
                            f"--iternum={iternum}", f"--num_cores={num_cores}", f"--generate_initial={generate_initial}",
                            f"--mapNpzFile={constantMapNpz}", f"--bdNpzFolder={constantMapAndBDFolder}",
                            f"--pathNpzFolders={LE}/iter{iternum}/eecbs_npzs"])
        print(command)
        subprocess.run(command, shell=True, check=True)
        # pdb.set_trace()

        # subprocess.run(["python", "./gnn/trainer.py", f"--exp_folder=./{LE}", f"--experiment=exp{expnum}", f"--iternum={iternum}", f"--num_cores={num_cores}", f"--generate_initial={generate_initial}"])
        log_time(f"Iter {iternum}: trainer")
        # run cs-pibt new maps to create new scenes
        # tmp = ["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", f"--iternum={iternum}", f"--num_samples={num_samples}", f"--max_samples={max_samples}"]
        # subprocess.run(["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", f"--iternum={iternum}", f"--num_samples={num_samples}", f"--max_samples={max_samples}"])
        command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={iterFolder}/eecbs_outputs", 
                        f"--num_parallel_runs={min(20, args.num_parallel)}",
                        "\"pymodel\"",
                        f"--modelPath={iterFolder}/models/max_test_acc.pt",
                        "--useGPU=False",
                        "--k=4",
                        "--m=5",
                        "--maxSteps=200"])
        first_iteration = "false"
        log_time(f"Iter {iternum}: simulator")
        
        # feed failures into eecbs
        iternum+=1
        command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
                        f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                        f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                        f"--outputFolder={nextIterFolder}/eecbs_outputs", 
                        f"--num_parallel_runs={args.num_parallel}",
                        "\"eecbs\"",
                        f"--outputPathNpzFolder={nextIterFolder}/eecbs_npzs",
                        f"--firstIter={first_iteration}",
                        f"--cutoffTime=20",
                        f"--suboptimality=2"])
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
        log_time(f"Iter {iternum}: eecbs")