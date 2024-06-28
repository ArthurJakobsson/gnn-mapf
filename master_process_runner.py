import os
import subprocess
import argparse
import pdb
import shutil

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("expnum", help="experiment number", type=int)
    parser.add_argument('mini_test', type=lambda x: bool(str2bool(x)))
    args = parser.parse_args()
    expnum, mini_test = args.expnum, args.mini_test
    print(args.expnum)

    iternum = 0
    if mini_test:
        source_maps_scens = "./data_collection/data/mini_benchmark_data"
    else: 
        source_maps_scens = "./data_collection/data/benchmark_data"

    LE = f"data_collection/data/logs/EXP{expnum}"
    os.makedirs(f"./{LE}", exist_ok=False)
    os.makedirs(f"./{LE}/labels", exist_ok=False)
    os.makedirs(f"./{LE}/labels/raw", exist_ok=False)

    first_iteration = "true"

    subprocess.run(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens", f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}"])

    print("Done with first eecbs run")
    pdb.set_trace()


    while True:

        if not os.path.exists(f"./{LE}/iter{iternum}"):
            os.makedirs(f"./{LE}/iter{iternum}")

        # train the naive model
        subprocess.run(["python", "./gnn/trainer.py", f"--exp_folder=./{LE}", f"--experiment=exp{expnum}", f"--iternum={iternum}"])
        pdb.set_trace()

        # run cs-pibt new maps to create new scenes
        subprocess.run(["python", "./gnn/simulator.py", f"--exp_folder=./{LE}", f"--firstIter={first_iteration}",f"--source_maps_scens={source_maps_scens}", f"--iternum={iternum}"])
        first_iteration = "false"

        pdb.set_trace()
        # TODO adjust source map scens
        shutil.rmtree("./data_collection/eecbs/raw_data/")
        # os.mkdir("./data_collection/eecbs/raw_data/") # don't remove these
        os.mkdir("./data_collection/eecbs/raw_data/bd")
        os.mkdir("./data_collection/eecbs/raw_data/paths")

        # feed failures into eecbs
        subprocess.run(["python", "./data_collection/eecbs_batchrunner.py", f"--mapFolder={source_maps_scens}/maps", f"--scenFolder=./{LE}/iter{iternum}/encountered_scens/", f"--outputFolder=../{LE}/labels/raw/", f"--expnum={expnum}", f"--firstIter={first_iteration}"]) # TODO figure out where eecbs is outputting files
        iternum+=1
        pdb.set_trace()
    '''
    LE = logs/EXPNAME/
    Initial collection: eecbs_runner.py -inputFolder=benchmark_data/scens -outputFolder=LE/iter0/labels/
    For K in range(0, 1000000)
        trainer.py -inputFolder=LE/iterK/labels/ -outputFolder=LE/iterK/models/
        Note: We should later modify trainer.py to take in all examples from iter0 … iterK
        simulation.py -inputFolder=LE/iterK/models -outputFolder=LE/iterK/encountered_scens/
        eecbs_runner.py -inputFolder=LE/iterK/encountered_scens/ -outputFolder=LE/iter{k+1}/labels/
    '''