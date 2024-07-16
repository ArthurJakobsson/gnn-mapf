import os
import subprocess
import argparse
import pdb
import shutil
import multiprocessing

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

    if generate_initial:
        os.makedirs(f"./{LE}", exist_ok=False)
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
    

    print("Done with first eecbs run")
    # pdb.set_trace()

    while True:        
        if not os.path.exists(f"./{LE}/iter{iternum}"):
            os.makedirs(f"./{LE}/iter{iternum}")

        # train the naive model
        subprocess.run(["python", "./gnn/trainer.py", f"--exp_folder=./{LE}", f"--experiment=exp{expnum}", f"--iternum={iternum}", f"--num_cores={num_cores}", f"--generate_initial={generate_initial}"])

        # run cs-pibt new maps to create new scenes
        subprocess.run(["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", f"--iternum={iternum}", f"--num_samples={num_samples}", f"--max_samples={max_samples}"])
        first_iteration = "false"

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
    '''
    LE = logs/EXPNAME/
    Initial collection: eecbs_runner.py -inputFolder=benchmark_data/scens -outputFolder=LE/iter0/labels/
    For K in range(0, 1000000)
        trainer.py -inputFolder=LE/iterK/labels/ -outputFolder=LE/iterK/models/
        Note: We should later modify trainer.py to take in all examples from iter0 … iterK
        simulation.py -inputFolder=LE/iterK/models -outputFolder=LE/iterK/encountered_scens/
        eecbs_runner.py -inputFolder=LE/iterK/encountered_scens/ -outputFolder=LE/iter{k+1}/labels/
    '''