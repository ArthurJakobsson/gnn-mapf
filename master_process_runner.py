import os
import subprocess
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("expnum", help="experiment number", type=int)
    args = parser.parse_args()
    expnum = args.expnum
    print(args.expnum)

    iternum = 0

    LE = f"data/logs/EXP{expnum}"

    subprocess.run("python", "./data_collection/eecbs_batchrunner.py", "-inputFolder=../data/benchmark_data/scens", f"-outputFolder=../{LE}/iter{iternum}/labels/raw/")



    while True:

        # train the naive model
        subprocess.run("python", "./gnn/trainer.py", f"-folder=../{LE}/iter{iternum}", f"-experiment=exp{expnum}", f"-iter=iter{iternum}")

        # run on new maps

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