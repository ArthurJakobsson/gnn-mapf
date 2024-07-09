import os
import subprocess
import argparse
import pdb
import shutil
from concurrent.futures import ProcessPoolExecutor

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("expnum", help="experiment number", type=int)
    parser.add_argument('mini_test', type=lambda x: bool(str2bool(x)))
    parser.add_argument('generate_initial', type=lambda x: bool(str2bool(x)))
    args = parser.parse_args()
    expnum, mini_test, generate_initial = args.expnum, args.mini_test, args.generate_initial
    print(args.expnum)

    iternum = 0
    if mini_test:
        source_maps_scens = "./data_collection/data/mini_benchmark_data"
    else: 
        source_maps_scens = "./data_collection/data/benchmark_data"

    LE = f"data_collection/data/logs/EXP{expnum}"
    

    first_iteration = "true"

    if generate_initial:
        os.makedirs(f"./{LE}", exist_ok=False)
        os.makedirs(f"./{LE}/labels", exist_ok=False)
        os.makedirs(f"./{LE}/labels/raw", exist_ok=False)
        subprocess.run(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens", f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}"])
    else:
        assert (os.path.exists(f"./{LE}/labels/raw/train_warehouse_10_20_10_2_2_0.npz"))
    

    print("Done with first eecbs run")

    while True:        
        if not os.path.exists(f"./{LE}/iter{iternum}"):
            os.makedirs(f"./{LE}/iter{iternum}")

        # train the naive model
        subprocess.run(["python", "./gnn/trainer.py", f"--exp_folder=./{LE}", f"--experiment=exp{expnum}", f"--iternum={iternum}"])

        # run cs-pibt new maps to create new scenes
        subprocess.run(["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", f"--iternum={iternum}"])
        first_iteration = "false"

        # feed failures into eecbs
        iternum+=1
        # Note: scen's should be taken from previous iteration but npz should be saved this next iteration number (since npzs have an extra)
        
        # for each map, move the corresponding scens to an ephermeral folder that is deleted after the batchrunner runs
        map_folder = f"{source_maps_scens}/maps"
        maps = [f.split("/")[-1] for f in os.listdir(map_folder)]
        params = []
        for map_file in maps: # name of the map, without parent dir names

            scenFolder = f"./{LE}/iter{iternum-1}/encountered_scens"

            # make temp folder
            os.makedirs(f"{scenFolder}/{map_file}")

            # copy all relevant scen files to that directory
            for scenFile in os.listdir(scenFolder):
                if ".scen" in scenFile and map_file in scenFile:
                    shutil.copyfile(scenFile, f"{scenFolder}/{map_file}")

            # save eecbs command
            params.append(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder={scenFolder}/{map_file}/", f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}", f"--iter={iternum}"]) # TODO figure out where eecbs is outputting files

        # run eecbs in parallel
        with ProcessPoolExecutor() as pool:
            pool.map(subprocess.run, params)

    '''
    LE = logs/EXPNAME/
    Initial collection: eecbs_runner.py -inputFolder=benchmark_data/scens -outputFolder=LE/iter0/labels/
    For K in range(0, 1000000)
        trainer.py -inputFolder=LE/iterK/labels/ -outputFolder=LE/iterK/models/
        Note: We should later modify trainer.py to take in all examples from iter0 … iterK
        simulation.py -inputFolder=LE/iterK/models -outputFolder=LE/iterK/encountered_scens/
        eecbs_runner.py -inputFolder=LE/iterK/encountered_scens/ -outputFolder=LE/iter{k+1}/labels/
    '''