import os
import argparse
import subprocess  # For executing eecbs script
import pandas as pd  # For smart batch running
import pdb # For debugging
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For utils
from collections import defaultdict
import shutil

import multiprocessing
# from custom_utils.multirunner import createTmuxSession, runCommandWithTmux, killTmuxSession

###############################################################
def createTmuxSession(i):
    tmux_session_name = f"worker_{i}"
    tmux_command = f"tmux new-session -d -s {tmux_session_name}"
    subprocess.run(tmux_command, shell=True, check=True)

def runCommandWithTmux(i, command):
    tmux_session_name = f"worker_{i}"
    new_command = f"{command}; tmux wait-for -S {tmux_session_name}_done"
    subprocess.run(["tmux", "send-keys", "-t", tmux_session_name, new_command, "C-m"], 
                   check=True)
    subprocess.run(["tmux", "wait-for", tmux_session_name + "_done"], check=True)
    # subprocess.run(["tmux", "kill-session", "-t", tmux_session_name], check=True)

def killTmuxSession(i):
    tmux_session_name = f"worker_{i}"
    subprocess.run(["tmux", "kill-session", "-t", tmux_session_name], check=True)
###############################################################

mapsToMaxNumAgents = { #TODO change this to 100 for all
    "Berlin_1_256": 101,
    "lt_gallowstemplar_n": 101, 
    "final_test9": 101,
    "empty_8_8": 20,
    "den312d": 500, 
    "final_test2": 101,
    "final_test6": 101,
    "random_32_32_10": 461,
    "brc202d": 101,
    "room_64_64_8": 101,
    "maze_128_128_1": 101,
    "warehouse_20_40_10_2_2": 101,
    "final_test3": 101,
    "Paris_1_256": 500,
    "final_test8": 101,
    "maze_128_128_10": 101, 
    "w_woundedcoast": 101,
    "maze_32_32_4": 101,
    "maze_32_32_2": 101,
    "ht_chantry": 500,
    "final_test1": 101,
    "empty_48_48": 500,
    "random_64_64_20": 101,
    "room_64_64_16": 101,
    "final_test4": 101,
    "empty_32_32": 511,
    "final_test7": 101,
    "Boston_0_256": 101,
    "random_64_64_10": 101,
    "empty_16_16": 101,
    "warehouse_10_20_10_2_2": 101,
    "room_32_32_4": 101,
    "final_test5": 101,
    "warehouse_10_20_10_2_1": 101,
    "warehouse_20_40_10_2_1": 101, 
    "ht_mansion_n": 101, 
    "maze_128_128_2": 101,
    "ost003d": 101,
    "orz900d": 101,
    "final_test0": 101,
    "random_32_32_20": 409,
    "lak303d": 101,
    "den520d": 500
}

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def runOnSingleInstance(eecbsArgs, outputfile, mapfile, numAgents, scenfile, 
                        runOrReturnCommand="run"):
    scenname = (scenfile.split("/")[-1])
    mapname = mapfile.split("/")[-1].split(".")[0]
    bd_path = f".{file_home}/eecbs/raw_data/{mapname}/bd/{scenname}{numAgents}.txt"
    # shutil.move(os.path.join(f".{file_home}/eecbs/raw_data/bd/", file_name), f".{file_home}/eecbs/raw_data/{mapFile}/bd/{file_name}") 
    assert(runOrReturnCommand in ["run", "return"])
    command = f".{file_home}/eecbs/build_release2/eecbs"

    for aKey in eecbsArgs:
        command += " --{}={}".format(aKey, eecbsArgs[aKey])
    # tempOutPath = f".{file_home}/eecbs/raw_data/paths/{scenname}{numAgents}.txt"
    tempOutPath = f".{file_home}/eecbs/raw_data/{mapname}/paths/{scenname}{numAgents}.txt"
    # shutil.move(os.path.join(f".{file_home}/eecbs/raw_data/paths/", file_name), f".{file_home}/eecbs/raw_data/{mapFile}/paths/{file_name}") 
    command += " --agentNum={} --agents={} --outputPaths={} --firstIter={} --bd_file={}".format(
                numAgents, scenfile, tempOutPath, firstIter, bd_path)
    # command += " --agentNum={} --seed={} --agentsFile={}".format(numAgents, seed, scenfile)
    command += " --output={} --map={}".format(outputfile, mapfile)
    # print(command.split("suboptimality=1")[1])
    if runOrReturnCommand == "run":
        subprocess.run(command.split(" "), check=True) # True if want failure error
    else:
        return command
    

def detectExistingStatus(eecbsArgs, mapfile, aNum, scenfile, df): # TODO update
    """
    Output:
        If has been run before
        Success if run before
    """
    if isinstance(df, str):
        if not os.path.exists(df):
            return False, 0
        df = pd.read_csv(df, index_col=False)  # index_col=False to avoid adding an extra index column
    # print(df)
    assert(isinstance(df, pd.DataFrame))

    ### Checks if the corresponding runs in the df have been completed already
    for aKey, aValue in eecbsArgs.items():
        if aKey == "output":
            continue
        if aKey not in df.columns:
            raise KeyError("Error: {} not in the columns of the dataframe".format(aKey))
        df = df[df[aKey] == aValue]  # Filter the dataframe to only include the runs with the same parameters
    df = df[(df["map"] == mapfile) & (df["agents"] == scenfile) & (df["agentNum"] == aNum)]
    if len(df) > 0:
        assert(len(df) == 1)
        success = df["solution cost"].values[0] != -1
        # success = df["overall_solution"].values[0] == 1
        return True, success
    else:
        return False, 0


####### Multi test
def runSingleInstanceMT(queue, nameToNumRun, lock, worker_id, idToWorkerOutputCSV, static_dict, 
                        eecbsArgs, mapName, curAgentNum, scen):
    # combinedDf = None
    workerOutputCSV = idToWorkerOutputCSV(worker_id, mapName)
    # combinedDf = workerOutputCSV
    # combined_filename = "data/logs/multiTest/{}_combined.csv".format(mapName)
    # combined_filename = f".{file_home}/eecbs/raw_data/{mapName}/csvs/combined.csv"
    combined_filename = idToWorkerOutputCSV(-1, mapName) # -1 denotes combined
    mapFile = static_dict[mapName]["map"]
    # eecbsArgs["output"] = workerOutputCSV

    runBefore, status = detectExistingStatus(eecbsArgs, mapFile, curAgentNum, scen, combined_filename) # check in combinedDf
    if not runBefore:
        command = runOnSingleInstance(eecbsArgs, workerOutputCSV, mapFile, curAgentNum, scen, runOrReturnCommand="return")
        # print(command)
        runCommandWithTmux(worker_id, command)
        runBefore, status = detectExistingStatus(eecbsArgs, mapFile, curAgentNum, scen, workerOutputCSV)  # check in worker's df
        if not runBefore:
            # print(f"worker_id:{worker_id}")
            print(f"mapFile:{mapFile}, curAgentNum:{curAgentNum}, scen:{scen}, workerOutputCSV:{workerOutputCSV}, worker_id:{worker_id}")
            raise RuntimeError("Fix detectExistingStatus; we ran an instance but cannot find it afterwards!")

    ## Update the number of runs. If we are the last one, then check if we should run the next agent number.
    lock.acquire()
    nameToNumRun[mapName] -= 1
    assert(nameToNumRun[mapName] >= 0)
    if nameToNumRun[mapName] == 0:
        queue.put(("checkIfRunNextAgents", (eecbsArgs, mapName, curAgentNum)))
    lock.release()


def checkIfRunNextAgents(queue, nameToNumRun, lock, num_workers, idToWorkerOutputCSV, static_dict, eecbsArgs, mapName, curAgentNum):
    ## Load separate CSVs and combine them
    # fileprefix = eecbsArgs["output"]
    combined_dfs = []
    for i in range(num_workers):
        filename = idToWorkerOutputCSV(i, mapName)
        if os.path.exists(filename):
            combined_dfs.append(pd.read_csv(filename, index_col=False))
    # combined_filename = "data/logs/multiTest/{}_combined.csv".format(mapName)
    # combined_filename = f".{file_home}/eecbs/raw_data/csvs/{mapName}_combined.csv"
    combined_filename = idToWorkerOutputCSV(-1, mapName) # -1 denotes combined
    if os.path.exists(combined_filename):
        combined_dfs.append(pd.read_csv(combined_filename, index_col=False))
    combined_df = pd.concat(combined_dfs)
    combined_df = combined_df.drop_duplicates()
    print("Map: {}, agentNum: {}, combined_df: {}".format(mapName, curAgentNum, combined_df.shape))
    combined_df.to_csv(combined_filename, index=False)
    
    ## Check status on current runs
    scens = static_dict[mapName]["scens"]
    agentNumsToRun = static_dict[mapName]["agentNumbers"]
    mapFile = static_dict[mapName]["map"]
    numSuccess = 0
    numToRunTotal = len(scens)
    for scen in scens:
        runBefore, status = detectExistingStatus(eecbsArgs, mapFile, curAgentNum, scen, combined_df)
        assert(runBefore)
        numSuccess += status

    if numSuccess < numToRunTotal/2:
        print("Early terminating as only succeeded {}/{} for {} agents on map {}".format(
                                        numSuccess, numToRunTotal, curAgentNum, mapName))
    else:
        assert(curAgentNum in agentNumsToRun)
        if agentNumsToRun.index(curAgentNum) + 1 == len(agentNumsToRun):
            print("Finished all agent numbers for map: {}".format(mapName))
        else:
            nextAgentNum = agentNumsToRun[agentNumsToRun.index(curAgentNum) + 1]
            lock.acquire()
            nameToNumRun[mapName] = len(scens)
            lock.release()
            for scen in scens:
                queue.put(("runSingleInstanceMT", (eecbsArgs, mapName, nextAgentNum, scen)))


def worker(queue: multiprocessing.JoinableQueue, nameToNumRun, lock,
            worker_id: int, num_workers: int, static_dict, idToWorkerOutputCSV):
    while True:
        task = queue.get()
        if task is None:
            break
        func, args = task
        if func == "checkIfRunNextAgents":
            checkIfRunNextAgents(queue, nameToNumRun, lock, num_workers, idToWorkerOutputCSV, static_dict, *args)
        elif func == "runSingleInstanceMT":
            runSingleInstanceMT(queue, nameToNumRun, lock, worker_id, idToWorkerOutputCSV, static_dict, *args)
        else:
            raise ValueError("Unknown function")
        queue.task_done()


def eecbs_runner(args):
    """
    Compare the runtime of EECBS and W-EECBS
    """
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    queue = multiprocessing.JoinableQueue()
    num_workers = args.num_parallel_runs
    workers = []

    ### Collect scens for each map
    mapsToScens = defaultdict(list)
    for dir_path in os.listdir(scenInputFolder):
        mapFile = dir_path.split("-")[0]
        if mapFile == ".DS_Store":
            continue 
        mapsToScens[mapFile].append(scenInputFolder+"/"+dir_path)

    ### For each map, get the static information
    static_dict = dict()
    nameToNumRun = manager.dict()
    for mapFile in mapsToScens:
        static_dict[mapFile] = {
            "map": f"{mapsInputFolder}/{mapFile}.map",
            "scens": mapsToScens[mapFile],
            # "agentNumbers": list(range(100, mapsToMaxNumAgents[mapFile]+1, 100)),
        }

        if "benchmark" in scenInputFolder: # pre-loop run
            increment = min(100,  mapsToMaxNumAgents[mapFile]-1)
            agentNumbers = list(range(increment, mapsToMaxNumAgents[mapFile]+1, increment))
            static_dict[mapFile]["agentNumbers"] = agentNumbers
            ### Run baseline EECBS
            # runOnSingleMap(eecbsArgs, mapFile, agentNumbers, scens, scenInputFolder)
        else: # we are somewhere in the training loop
            agentNumbers = []
            scenlist = os.listdir(scenInputFolder)
            for scen in scenlist:
                # open the file
                with open(f'{scenInputFolder}/{scen}', 'r') as fh: # TODO get the path to scene file right
                    line = fh.readline()
                    agentNumbers.append(int(line))
            static_dict[mapFile]["agentNumbers"] = agentNumbers
            # run eecbs
            # runOnSingleMap(eecbsArgs, mapFile, agentNumbers, scens, scenInputFolder)
        
        nameToNumRun[mapFile] = len(mapsToScens[mapFile])


    os.makedirs(f".{file_home}/eecbs/raw_data/csvs/", exist_ok=True)
    def idToWorkerOutputCSV(worker_id, mapName):
        assert(worker_id >= -1)
        if worker_id == -1: # worker_id = -1 denotes combined csv
            return f".{file_home}/eecbs/raw_data/{mapName}/csvs/combined.csv"
        return f".{file_home}/eecbs/raw_data/{mapName}/csvs/worker_{worker_id}.csv"

    # Start worker processes with corresponding tmux session
    for worker_id in range(num_workers):
        createTmuxSession(worker_id)
        # workerOutputCSV = "{}/worker_{}.csv".format(logPath, worker_id)
        p = multiprocessing.Process(target=worker, 
                    args=(queue, nameToNumRun, lock, worker_id, num_workers, static_dict, idToWorkerOutputCSV))
        p.start()
        workers.append(p)

    eecbsArgs = {
        "suboptimality": args.suboptimality,
        "cutoffTime": args.cutoffTime,
        # "firstIter": args.firstIter,
    }
    # pdb.set_trace()

    ## Create initial jobs
    for mapFile in mapsToScens.keys():
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}", exist_ok=True)
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/bd/", exist_ok=True)
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/paths/", exist_ok=True)
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/csvs/", exist_ok=True)
        for scen in static_dict[mapFile]["scens"]:
            queue.put(("runSingleInstanceMT", (eecbsArgs, mapFile, static_dict[mapFile]["agentNumbers"][0], scen)))

    # Wait for all tasks to be processed
    queue.join()

    # Stop worker processes
    for i in range(num_workers):
        killTmuxSession(i)
        queue.put(None)
    for p in workers:
        p.join()

    # Delete CSV files with {mapName}_worker naming convention
    for mapName in mapsToScens.keys():
        for worker_id in range(num_workers):
            filename = idToWorkerOutputCSV(worker_id, mapName)
            if os.path.exists(filename):
                os.remove(filename)

    # print("All tasks completed!")

    # pdb.set_trace()
    for mapFile in mapsToScens:
        # move the new eecbs output
        # os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}", exist_ok=True)
        # os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/bd/", exist_ok=True)
        # os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/paths/", exist_ok=True)

        # bd_files = os.listdir(f".{file_home}/eecbs/raw_data/bd/")
        # # path_files = os.listdir(f".{file_home}/eecbs/raw_data/paths/")
        # for file_name in bd_files:
        #     shutil.move(os.path.join(f".{file_home}/eecbs/raw_data/bd/", file_name), f".{file_home}/eecbs/raw_data/{mapFile}/bd/{file_name}") 
        # for file_name in path_files:
        #     shutil.move(os.path.join(f".{file_home}/eecbs/raw_data/paths/", file_name), f".{file_home}/eecbs/raw_data/{mapFile}/paths/{file_name}") 
        
        # shutil.move(f".{file_home}/eecbs/raw_data/bd/", f".{file_home}/eecbs/raw_data/{mapFile}/bd/")
        # shutil.move(f".{file_home}/eecbs/raw_data/paths/", f".{file_home}/eecbs/raw_data/{mapFile}/paths/")
        # os.makedirs(f".{file_home}/eecbs/raw_data/paths/", exist_ok=True)

        # make the npz files
        # TODO don't hardcode exp, iter numbers
        subprocess.run(["python", "./data_collection/data_manipulator.py", f"--pathsIn=.{file_home}/eecbs/raw_data/{mapFile}/paths/", 
                        f"--bdIn=.{file_home}/eecbs/raw_data/{mapFile}/bd/", 
                        f"--mapIn={mapsInputFolder}", f"--trainOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/train_{mapFile}_{args.iter}", 
                        f"--valOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/val_{mapFile}_{args.iter}"])
        # pdb.set_trace()
        # subprocess.run(["python", "./data_collection/data_manipulator.py", f"--pathsIn=.{file_home}/eecbs/raw_data/{mapFile}/paths/", 
        #                 f"--bdIn=.{file_home}/eecbs/raw_data/{mapFile}/bd/", 
        #                 f"--mapIn={mapsInputFolder}", f"--trainOut=.{file_home}/eecbs/raw_data/train_{mapFile}_{args.iter}", 
        #                 f"--valOut=.{file_home}/eecbs/raw_data/train_{mapFile}_{args.iter}"])
        # pdb.set_trace()

        # os.mkdir(f".{file_home}/eecbs/raw_data/bd")
        # os.mkdir(f".{file_home}/eecbs/raw_data/paths")
        # shutil.rmtree(f".{file_home}/eecbs/raw_data/{mapFile}/paths") # clean path files once they have been recorded
        # os.mkdir(f".{file_home}/eecbs/raw_data/{mapFile}/paths") # remake "path" folder

# python batch_runner.py den312d --logPath data/logs/test --cutoffTime 10 --suboptimality 2
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mapFolder", help="contains all scens to run", type=str)
    parser.add_argument("--scenFolder", help="contains all scens to run", type=str)
    parser.add_argument("--outputFolder", help="place to output all eecbs output", type=str)
    parser.add_argument("--cutoffTime", help="cutoffTime", type=int, default=60)
    parser.add_argument("--suboptimality", help="suboptimality", type=float, default=2)
    parser.add_argument("--expnum", help="experiment number", type=int, default=0)
    parser.add_argument("--iter", help="iteration number", type=int)
    parser.add_argument('--firstIter', dest='firstIter', type=lambda x: bool(str2bool(x)))
    parser.add_argument('--num_parallel_runs', help="How many multiple maps in parallel tmux sessions. 1 = No parallel runs.", 
                        type=int)
    args = parser.parse_args()
    scenInputFolder = args.scenFolder
    mapsInputFolder = args.mapFolder
    firstIter = args.firstIter
    file_home = "/data_collection"
    print("iternum: " + str(args.iter))
    eecbs_runner(args)