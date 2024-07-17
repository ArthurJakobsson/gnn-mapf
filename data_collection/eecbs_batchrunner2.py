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
# from custom_utils.custom_timer import CustomTimer
from custom_utils.custom_timer import CustomTimer
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

def runOnSingleInstance(eecbsArgs, outputFolder, outputfile, mapfile, numAgents, scenfile, 
                        runOrReturnCommand="run"):
    scenname = (scenfile.split("/")[-1])
    mapname = mapfile.split("/")[-1].split(".")[0]
    bd_path = f"{outputFolder}/bd/{scenname}{numAgents}.txt"
    assert(runOrReturnCommand in ["run", "return"])
    # command = f".{file_home}/eecbs/build_release2/eecbs"
    command = f"{eecbsPath}"

    for aKey in eecbsArgs:
        command += " --{}={}".format(aKey, eecbsArgs[aKey])
    tempOutPath = f"{outputFolder}/paths/{scenname}{numAgents}.txt"
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
    workerOutputCSV = idToWorkerOutputCSV(worker_id, mapName)
    combined_filename = idToWorkerOutputCSV(-1, mapName) # -1 denotes combined
    mapFile = static_dict[mapName]["map"]
    outputFolder = static_dict[mapName]["outputFolder"]

    runBefore, status = detectExistingStatus(eecbsArgs, mapFile, curAgentNum, scen, combined_filename) # check in combinedDf
    if not runBefore:
        command = runOnSingleInstance(eecbsArgs, outputFolder, workerOutputCSV, mapFile, curAgentNum, scen, runOrReturnCommand="return")
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
    # scens = static_dict[mapName]["scens"]
    # agentNumsToRun = static_dict[mapName]["agentNumbers"]
    mapFile = static_dict[mapName]["map"]
    numSuccess = 0
    numToRunTotal = len(static_dict[mapName]["scens"])
    for scen, agentNum in zip(static_dict[mapName]["scens"], static_dict[mapName]["agentsPerScen"]):
        runBefore, status = detectExistingStatus(eecbsArgs, mapFile, curAgentNum, scen, combined_df)
        assert(runBefore)
        numSuccess += status

    # shouldRunDataManipulator = False
    if len(static_dict[mapName]["agentRange"]) > 0: # If we are running a range of agents, check the next one
        if numSuccess < numToRunTotal/2:
            print("Early terminating as only succeeded {}/{} for {} agents on map {}".format(
                                            numSuccess, numToRunTotal, curAgentNum, mapName))
            # shouldRunDataManipulator = True
        else:
            scens = static_dict[mapName]["scens"]
            agentNumsToRun = static_dict[mapName]["agentRange"]
            assert(curAgentNum in agentNumsToRun)
            if agentNumsToRun.index(curAgentNum) + 1 == len(agentNumsToRun):
                print("Finished all agent numbers for map: {}".format(mapName))
                # shouldRunDataManipulator = True
            else:
                nextAgentNum = agentNumsToRun[agentNumsToRun.index(curAgentNum) + 1]
                lock.acquire()
                nameToNumRun[mapName] = len(scens)
                lock.release()
                static_dict[mapName]["agentsPerScen"] = [nextAgentNum] * len(scens)
                for scen, agentNum in zip(scens, static_dict[mapName]["agentsPerScen"]):
                    queue.put(("runSingleInstanceMT", (eecbsArgs, mapName, agentNum, scen)))
    else: # If not, we are done
        print("Finished all agent numbers for map: {}, succeeded {}/{}".format(mapName, numSuccess, numToRunTotal))
        # shouldRunDataManipulator = True
    
    # if shouldRunDataManipulator:
    #     queue.put(("runDataManipulator", 
    #                 (mapName, )))
                        # static_dict[mapName]["scens"], f".{file_home}/eecbs/raw_data/{mapName}/bd/", mapFile, 
                        # f".{file_home}/data/logs/EXP{args.expnum}/labels/raw/train_{mapName}_{args.iter}", 
                        # f".{file_home}/data/logs/EXP{args.expnum}/labels/raw/val_{mapName}_{args.iter}")))


# def runDataManipulator(worker_id, mapName):
#     # print("AT DATA MANIPULATOR")
#     trainOut = f".{file_home}/data/logs/EXP{args.expnum}/labels/raw/train_{mapName}_{args.iter}"
#     valOut = f".{file_home}/data/logs/EXP{args.expnum}/labels/raw/val_{mapName}_{args.iter}"
#     if os.path.exists(trainOut) and os.path.exists(valOut):
#         print("Data already exists")
#         return

#     # command = ["python", "./data_collection/data_manipulator.py", 
#     command = ["python", "-m", "data_collection.data_manipulator", 
#                 f"--pathsIn=.{file_home}/eecbs/raw_data/{mapName}/paths/", 
#                 f"--bdIn=.{file_home}/eecbs/raw_data/{mapName}/bd/", 
#                 f"--mapIn={mapsInputFolder}", f"--trainOut={trainOut}", 
#                 f"--valOut={valOut}", "--num_parallel=3"]
#     # os.get
#     runCommandWithTmux(worker_id, " ".join(command))
    


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
        # elif func == "runDataManipulator":
        #     runDataManipulator(worker_id, *args)
        else:
            raise ValueError("Unknown function")
        queue.task_done()

def helperRun(command):
    subprocess.run(command, check=True, shell=True)

def eecbs_runner(args):
    """
    Compare the runtime of EECBS and W-EECBS
    """
    ### Start filesystem logic
    outputFolder = args.outputFolder
    os.makedirs(outputFolder, exist_ok=True)
    scenInputFolder = args.scenFolder
    mapsInputFolder = args.mapFolder
    global firstIter, eecbsPath
    firstIter = args.firstIter
    eecbsPath = args.eecbsPath
    assert(eecbsPath.startswith(".") and eecbsPath.endswith("eecbs") and os.path.exists(eecbsPath))

    
    ### Start multiprocessing logic
    ct = CustomTimer()
    ct.start("EECBS Calls")
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
    # This includes the map file, the scens, and the number of agents to run
    static_dict = dict()
    nameToNumRun = manager.dict()
    for mapFile in mapsToScens: # mapFile is just the map name without the path or .map extension
        static_dict[mapFile] = {
            "map": f"{mapsInputFolder}/{mapFile}.map",
            "scens": mapsToScens[mapFile],
            "agentsPerScen": [], # This is used for when each scen has a specific number of agents already specified, e.g. for encountered scens
            "agentRange": [], # This is used for when we want to run a range of agents, e.g. for benchmark scens
            "outputFolder": f"{outputFolder}/{mapFile}", # This is where the output of the runs
        }

        if "benchmark" in scenInputFolder: # pre-loop run
            increment = min(100,  mapsToMaxNumAgents[mapFile]-1)
            agentNumbers = list(range(increment, mapsToMaxNumAgents[mapFile]+1, increment))
            static_dict[mapFile]["agentRange"] = agentNumbers
            # pdb.set_trace()
            static_dict[mapFile]["agentsPerScen"] = [agentNumbers[0]] * len(static_dict[mapFile]["scens"])
        else: # we are somewhere in the training loop
            agentNumbers = []
            scenlist = os.listdir(scenInputFolder)
            for scen in scenlist:
                # open the file
                with open(f'{scenInputFolder}/{scen}', 'r') as fh: # TODO get the path to scene file right
                    firstLine = fh.readline()
                    assert(firstLine.startswith("version "))
                    num = firstLine.split(" ")[1]
                    agentNumbers.append(int(num))
            static_dict[mapFile]["agentsPerScen"] = agentNumbers
        
        assert(len(static_dict[mapFile]["agentsPerScen"]) > 0)
        nameToNumRun[mapFile] = len(mapsToScens[mapFile])
        # print(f"Map: {mapFile} requires {nameToNumRun[mapFile]} runs")


    def idToWorkerOutputCSV(worker_id, mapName):
        assert(worker_id >= -1)
        if worker_id == -1: # worker_id = -1 denotes combined csv
            return f"{outputFolder}/{mapName}/csvs/combined.csv"
        return f"{outputFolder}/{mapName}/csvs/worker_{worker_id}.csv"

    # pdb.set_trace()
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
        mapOutputFolder = static_dict[mapFile]["outputFolder"]
        # print(mapOutputFolder)
        os.makedirs(mapOutputFolder, exist_ok=True)
        os.makedirs(f"{mapOutputFolder}/bd/", exist_ok=True)
        os.makedirs(f"{mapOutputFolder}/paths/", exist_ok=True)
        os.makedirs(f"{mapOutputFolder}/csvs/", exist_ok=True)
        # pdb.set_trace()
        for scen, agentNum in zip(static_dict[mapFile]["scens"], static_dict[mapFile]["agentsPerScen"]):
            queue.put(("runSingleInstanceMT", (eecbsArgs, mapFile, agentNum, scen)))

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
    ct.stop("EECBS Calls")
    print("------------ Finished EECBS Calls in :{:.3f} seconds".format(ct.getTimes("EECBS Calls")))

    # pdb.set_trace()
    ### Create folder for saving data_manipulator npz files
    outputPathNpzFolder = args.outputPathNpzFolder
    if outputPathNpzFolder is None:
        outputPathNpzFolder = f"{outputFolder}/../eecbs_npzs"
    os.makedirs(outputPathNpzFolder, exist_ok=True)

    ### Run data manipulator with multiprocessing
    ct.start("Data Manipulator")
    input_list = []
    numWorkersParallelForDataManipulator = 1
    for mapFile in mapsToScens: # mapFile is just the map name without the path or .map extension
        mapOutputFolder = static_dict[mapFile]["outputFolder"]
        pathsIn = f"{mapOutputFolder}/paths/"
        if not os.listdir(pathsIn): # if the pathsIn folder is empty
            print("Skipping: {} as all failed".format(mapFile))
            continue
        # mapOutputNpz = f".{file_home}/data/benchmark_data/constant_npzs/{mapFile}_map.npz"
        # bdOutputNpz = f".{file_home}/data/benchmark_data/constant_npzs/{mapFile}_bds.npz"
        mapOutputNpz = f"{args.constantMapAndBDFolder}/{mapFile}_map.npz"
        bdOutputNpz = f"{args.constantMapAndBDFolder}/{mapFile}_bds.npz"

        command = " ".join(["python", "-m", "data_collection.data_manipulator", 
                        f"--pathsIn={pathsIn}", f"--pathOutFile={outputPathNpzFolder}/{mapFile}_paths.npz",
                        f"--bdIn={mapOutputFolder}/bd", f"--bdOutFile={bdOutputNpz}", 
                        f"--mapIn={mapsInputFolder}", f"--mapOutFile={args.constantMapAndBDFolder}/all_maps.npz", #f"--mapOutFile={mapOutputNpz}",
                        # f"--trainOut={outputFolder}/eecbs_npz/train_{mapFile}_{args.iter}",
                        # f"--trainOut={file_home}/data/logs/EXP{args.expnum}/labels/raw/train_{mapFile}_{args.iter}", 
                        # f"--valOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/val_{mapFile}_{args.iter}",
                        f"--num_parallel={numWorkersParallelForDataManipulator}"])
        input_list.append((command,))
        # make the npz files
        # TODO don't hardcode exp, iter numbers
        # subprocess.run(["python", "./data_collection/data_manipulator.py", f"--pathsIn=.{file_home}/eecbs/raw_data/{mapFile}/paths/", 
        #                 f"--bdIn=.{file_home}/eecbs/raw_data/{mapFile}/bd/", 
        #                 f"--mapIn={mapsInputFolder}", f"--trainOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/train_{mapFile}_{args.iter}", 
        #                 f"--valOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/val_{mapFile}_{args.iter}"])
    # pdb.set_trace()
    with multiprocessing.Pool(processes=min(len(mapName), num_workers//numWorkersParallelForDataManipulator)) as pool:
        pool.starmap(helperRun, input_list)
    ct.stop("Data Manipulator")
    print("------------ Finished Data Manipulator in :{:.3f} seconds".format(ct.getTimes("Data Manipulator")))
    


# python ./data_collection/data_manipulator.py --pathsIn=data_collection/eecbs/raw_data/den520d/paths --bdIn=data_collection/eecbs/raw_data/den520d/bd/ 
# --mapIn=data_collection/data/benchmark_data/maps/ --trainOut=data_collection/data/logs/EXP0/labels/raw/train_den520d_0 --valOut=data_collection/data/logs/EXP0/labels/raw/val_den520d_0

### Example call from master_process_runner.py
# python -m data_collection.eecbs_batchrunner2 --mapFolder=./data_collection/data/benchmark_data/maps --scenFolder=./data_collection/data/benchmark_data/scens 
#                   --outputFolder=../data_collection/data/logs/EXP0/labels/raw/ --expnum=0 --firstIter=true --iter=0 --num_parallel=10 --cutoffTime=20
### Updated one
# python -m data_collection.eecbs_batchrunner2 --mapFolder=data_collection/data/benchmark_data/maps --scenFolder=data_collection/data/benchmark_data/scens 
#                  --constantMapAndBDFolder=data_collection/data/benchmark_data/constant_npzs/ --outputPathNpzFolder=data_collection/data/logs/EXP_Test/iter0/eecbs_npzs
#                  --outputFolder=data_collection/data/logs/EXP_Test/iter0/eecbs_outputs --expnum=0 --firstIter=true --iter=0 --num_parallel=10 --cutoffTime=5
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mapFolder", help="contains all scens to run", type=str)
    parser.add_argument("--scenFolder", help="contains all scens to run", type=str)
    parser.add_argument("--constantMapAndBDFolder", help="contains the precomputed map and bd folders", type=str)
    parser.add_argument("--outputFolder", help="place to output all eecbs output, parent folder of each map folder", type=str)
    parser.add_argument("--outputPathNpzFolder", help="folder for data_manipulator to create npz files", type=str, default=None)
    parser.add_argument("--eecbsPath", help="path to eecbs executable", type=str, default="./data_collection/eecbs/build_release2/eecbs")
    parser.add_argument("--cutoffTime", help="cutoffTime", type=int, default=60)
    parser.add_argument("--suboptimality", help="suboptimality", type=float, default=2)
    parser.add_argument("--expnum", help="experiment number", type=int, default=0)
    parser.add_argument("--iter", help="iteration number", type=int)
    parser.add_argument('--firstIter', dest='firstIter', type=lambda x: bool(str2bool(x)))
    parser.add_argument('--num_parallel_runs', help="How many multiple maps in parallel tmux sessions. 1 = No parallel runs.", 
                        type=int)
    args = parser.parse_args()
    # scenInputFolder = args.scenFolder
    # mapsInputFolder = args.mapFolder
    # firstIter = args.firstIter
    # file_home = "/data_collection"
    print("iternum: " + str(args.iter))
    eecbs_runner(args)