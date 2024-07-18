import os
import subprocess
import argparse
import pdb
import shutil
import multiprocessing
from datetime import datetime

last_recorded_time = datetime.now()

def log_time(event_name):
    global last_recorded_time
    cur_time = datetime.now()
    with open(f"./timing_folder/master_timing.txt", mode='a') as file:
        file.write(f"{event_name} recorded at {cur_time}. \t\t Duration: \t {(cur_time-last_recorded_time).total_seconds()} \n")
    last_recorded_time  = cur_time

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

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

    LE = f"data_collection/data/logs/EXP{expnum}"
    os.makedirs(LE, exist_ok=True)
    
    num_cores = multiprocessing.cpu_count()
    first_iteration = "true"
    print("Current Path:", os.getcwd())

    # command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
    #                     f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
    #                     f"--num_parallel={args.num_parallel}", "--cutoffTime=20"])
    # print(command)
    command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens", 
                            f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
                            f"--num_parallel_runs={args.num_parallel}", "--cutoffTime=20"])
    print(command)
    # pdb.set_trace()
    
    log_time("begin")
    
    if generate_initial:
        os.makedirs(f"./{LE}/labels", exist_ok=False)
        os.makedirs(f"./{LE}/labels/raw", exist_ok=False)
        command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens", 
                            f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
                            f"--num_parallel_runs={args.num_parallel}", "--cutoffTime=20"])
        ### Example command for minibenchmark
        # python ./data_collection/eecbs_batchrunner2.py --mapFolder=./data_collection/data/mini_benchmark_data/maps 
        #           --scenFolder=./data_collection/data/mini_benchmark_data/scens --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ 
        #           --expnum=0 --firstIter=true --iter=0 --num_parallel=50 --cutoffTime=20
        ### Example command for benchmark
        # python ./data_collection/eecbs_batchrunner2.py --mapFolder=./data_collection/data/benchmark_data/maps 
        #           --scenFolder=./data_collection/data/benchmark_data/scens --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ 
        #           --expnum=0 --firstIter=true --iter=0 --num_parallel=50 --cutoffTime=20
        subprocess.run(command, shell=True, check=True)
        # print(command)
        # subprocess.run(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}"], check=True)
    
    else:
        assert (os.path.exists(f"./{LE}/labels/raw/train_warehouse_10_20_10_2_2_0.npz"))
    
    log_time("initial generation")
    print("Done with first eecbs run")
    # pdb.set_trace()

    while True:        
        if not os.path.exists(f"./{LE}/iter{iternum}"):
            os.makedirs(f"./{LE}/iter{iternum}")

        # train the naive model
        subprocess.run(["python", "./gnn/trainer.py", f"--exp_folder=./{LE}", f"--experiment=exp{expnum}", f"--iternum={iternum}", f"--num_cores={num_cores}", f"--generate_initial={generate_initial}"])
        log_time(f"training for iteration: {iternum}")
        # run cs-pibt new maps to create new scenes
        print((" ").join(["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", 
                        f"--iternum={iternum}", f"--num_samples={num_samples}", f"--max_samples={max_samples}",
                        f"--label_folder=./{LE}/labels/raw/"]))
        subprocess.run(["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", 
                        f"--iternum={iternum}", f"--num_samples={num_samples}", f"--max_samples={max_samples}",
                        f"--label_folder=./{LE}/labels/raw/"])
        first_iteration = "false"
        log_time(f"simulating for iteration: {iternum}")
        
        # feed failures into eecbs
        iternum+=1
        # Note: scen's should be taken from previous iteration but npz should be saved this next iteration number (since npzs have an extra)
        # subprocess.run(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}"]) # TODO figure out where eecbs is outputting files
        # command = " ".join(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
        #                 f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
        #                 f"--num_parallel={args.num_parallel}", "--cutoffTime=20"])
        subprocess.run(["python", "./data_collection/eecbs_batchrunner2.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum-1}/encountered_scens", 
                        f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}",
                        f"--num_parallel_runs={args.num_parallel}", "--cutoffTime=20"]) # TODO figure out where eecbs is outputting files
        log_time(f"eecbs running for iteration: {iternum}")