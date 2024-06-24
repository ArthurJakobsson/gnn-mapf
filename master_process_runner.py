import os
import subprocess
import argparse
import pdb

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

    subprocess.run(["python3", "./data_collection/eecbs_batchrunner.py", "--inputFolder=../data/benchmark_data/scens", f"--outputFolder=../{LE}/labels/raw/"])

    pdb.set_trace()
    raise NotImplementedError

    first_iteration = True

    while True:

        if not os.path.exists(f"./data_collection/data/logs/EXP{expnum}"):
            os.makedirs(f"./data_collection/data/logs/EXP{expnum}")

        # train the naive model
        subprocess.run(["python3", "./gnn/trainer.py", f"--folder=../{LE}/iter{iternum}", f"--experiment=exp{expnum}", f"--iternum={iternum}"])

        # run cs-pibt new maps to create new scenes
        subprocess.run(["python3", "./gnn/simulator.py", f"--folder=../{LE}/iter{iternum}", f"--firstIter={first_iteration}", f"--source_maps_scens={source_maps_scens}"])
        first_iteration = False

        # feed failures into eecbs
        subprocess.run(["python3", "./data_collection/eecbs_batchrunner.py"]) # TODO figure out where eecbs is outputting files
        iternum+=1
    '''
    LE = logs/EXPNAME/
    Initial collection: eecbs_runner.py -inputFolder=benchmark_data/scens -outputFolder=LE/iter0/labels/
    For K in range(0, 1000000)
        trainer.py -inputFolder=LE/iterK/labels/ -outputFolder=LE/iterK/models/
        Note: We should later modify trainer.py to take in all examples from iter0 … iterK
        simulation.py -inputFolder=LE/iterK/models -outputFolder=LE/iterK/encountered_scens/
        eecbs_runner.py -inputFolder=LE/iterK/encountered_scens/ -outputFolder=LE/iter{k+1}/labels/
    '''