import os
import argparse
import subprocess  # For executing eecbs script
import pandas as pd  # For smart batch running
import pdb # For debugging
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For utils
from collections import defaultdict
import shutil
import json
import glob


import ray
import ray.util.multiprocessing
# from custom_utils.custom_timer import CustomTimer
from custom_utils.custom_timer import CustomTimer
from custom_utils.common_helper import str2bool, getMapScenAgents


@ray.remote
class SharedDict:
    def __init__(self):
        self.dict = {}

    def set(self, key, val):
        self.dict[key] = val

    def decrement_get(self, key):
        self.dict[key] -= 1
        return self.dict[key]

    
mapsToMaxNumAgents = {
    "Berlin_1_256": 1000,
    "Boston_0_256": 1000,
    "Paris_1_256": 1000,
    "brc202d": 1000,
    "den312d": 1000, 
    "den520d": 1000,
    "dense_map_15_15_0":50,
    "dense_map_15_15_1":50,
    "corridor_30_30_0":50,
    "empty_8_8": 32,
    "empty_16_16": 128,
    "empty_32_32": 512,
    "empty_48_48": 1000,
    "ht_chantry": 1000,
    "ht_mansion_n": 1000,
    "lak303d": 1000,
    "lt_gallowstemplar_n": 1000,
    "maze_128_128_1": 1000,
    "maze_128_128_10": 1000,
    "maze_128_128_2": 1000,
    "maze_32_32_2": 333,
    "maze_32_32_4": 395,
    "orz900d": 1000,
    "ost003d": 1000,
    "random_32_32_10": 461,
    "random_32_32_10_custom_0": 461,
    "random_32_32_10_custom_1": 461,
    "random_32_32_20": 409,
    "random_64_64_10_custom_0": 1000,
    "random_64_64_10_custom_1": 1000,
    "random_64_64_10": 1000,
    "random_64_64_20": 1000,
    "room_32_32_4": 341,
    "room_64_64_16": 1000,
    "room_64_64_8": 1000,
    "w_woundedcoast": 1000,
    "warehouse_10_20_10_2_1": 1000,
    "warehouse_10_20_10_2_2": 1000,
    "warehouse_20_40_10_2_1": 1000,
    "warehouse_20_40_10_2_2": 1000,
}


def getEECBSCommand(eecbsArgs, outputFolder, outputfile, mapfile, numAgents, scenfile):
    """Command for running EECBS"""
    _, scenname, _ = getMapScenAgents(scenfile)
    bd_path = f"{outputFolder}/bd/{scenname}.{numAgents}.txt"
    command = f"{eecbsPath}"

    for aKey in eecbsArgs["args"]:
        command += " --{}={}".format(aKey, eecbsArgs["args"][aKey])
        
    command += " --agentNum={} --agents={} --firstIter={} --bd_file={}".format(
                numAgents, scenfile, firstIter, bd_path)
    command += " --output={} --map={}".format(outputfile, mapfile)
    command += " --sipp=1"
    return command
    

def getCommandForSingleInstance(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile):
    if runnerArgs["command"] == "eecbs":
        return getEECBSCommand(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile)
    else:
        raise ValueError("Unknown command: {}".format(runnerArgs["command"]))


def detectExistingStatus(runnerArgs, mapfile, aNum, scenfile, df): # TODO update
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

    ### Grabs the correct row from the dataframe based on arguments
    for aKey, aValue in runnerArgs["args"].items():
        if aKey == "extra_layers" or aKey == "bd_pred" or aKey=="timeLimit":
            continue
        if aKey not in df.columns:
            
            raise KeyError("Error: {} not in the columns of the dataframe".format(aKey))
        df = df[df[aKey] == aValue]  # Filter the dataframe to only include the runs with the same parameters
    
    df = df[(df["map"] == mapfile) & (df["agents"] == scenfile) & (df["agentNum"] == aNum)]
    
    ### Checks if the corresponding runs in the df have been completed already
    if len(df) > 0:
        # assert(len(df) == 1)
        if len(df) > 1:
            print("Warning, multiple runs with the same parameters, likely due to a previous crash")
            print("Map: {}, NumAgents: {}, Scen: {}, # Found: {}".format(mapfile, aNum, scenfile, len(df)))
        success = df["solution cost"].values[-1] != -1
        return True, success
    else:
        return False, 0


####### Multi test
@ray.remote
def runSingleInstanceMT(nameToNumRun, num_workers, idToWorkerOutputFilepath, static_dict,
                        runnerArgs, mapName, curAgentNum, scen):
    worker_id = ray.get_runtime_context().get_worker_id()

    workerOutputCSV = idToWorkerOutputFilepath(worker_id, mapName)
    combined_filename = idToWorkerOutputFilepath(-1, mapName) # -1 denotes combined
    mapFile = static_dict[mapName]["map"]
    outputFolder = static_dict[mapName]["outputFolder"]

    runBefore, status = detectExistingStatus(runnerArgs, mapFile, curAgentNum, scen, combined_filename) # check in combinedDf
    if not runBefore:
        command = getCommandForSingleInstance(runnerArgs, outputFolder, workerOutputCSV, mapFile, curAgentNum, scen)
        helperRun(command)
        runBefore, status = detectExistingStatus(runnerArgs, mapFile, curAgentNum, scen, workerOutputCSV)  # check in worker's df
        if not runBefore:
            pass
            # print(f"mapFile:{mapFile}, curAgentNum:{curAgentNum}, scen:{scen}, workerOutputCSV:{workerOutputCSV[:-54]}, worker_id:{worker_id[:-50]}")
            # raise RuntimeError("Fix detectExistingStatus; we ridToWorkerOutputFilepathan an instance but cannot find it afterwards!")

    ## Update the number of runs. If we are the last one, then check if we should run the next agent number.
    runsLeft = ray.get(nameToNumRun.decrement_get.remote(mapName))
    assert(runsLeft >= 0) # 1 run per scen when generating bds
    if runsLeft == 0:
        return [checkIfRunNextAgents.remote(nameToNumRun, num_workers, idToWorkerOutputFilepath, static_dict,
                                            runnerArgs, mapName, curAgentNum)]
    return []


@ray.remote
def checkIfRunNextAgents(nameToNumRun, num_workers, idToWorkerOutputFilepath, static_dict, 
                         runnerArgs, mapName, curAgentNum):
    """Inputs:
        curAgentNum: This is only used if static_dict["agentRange"] is not empty
    Output:
        futures: List of futures, empty if we are done
    """
    futures = []

    finished = False
    if len(static_dict[mapName]["agentRange"]) > 0:
        scens = static_dict[mapName]["scens"]
        agentNumsToRun = static_dict[mapName]["agentRange"]
        assert(curAgentNum in agentNumsToRun)
        if agentNumsToRun.index(curAgentNum) + 1 == len(agentNumsToRun):
            print("Finished all scens for map: {}".format(mapName))
            finished = True
        else:
            nextAgentNum = agentNumsToRun[agentNumsToRun.index(curAgentNum) + 1]
            nameToNumRun.set.remote(mapName, len(scens))
            static_dict[mapName]["agentsPerScen"] = [nextAgentNum] * len(scens)
            for scen, agentNum in zip(scens, static_dict[mapName]["agentsPerScen"]):
                args = (runnerArgs, mapName, agentNum, scen)
                futures.append(runSingleInstanceMT.remote(nameToNumRun, num_workers, idToWorkerOutputFilepath, 
                                                            static_dict, *args))
    else: # If not, we are done
        print("Finished all scens for map: {}".format(mapName))
        finished = True

    if finished:
        finished_txt = idToWorkerOutputFilepath(-2, mapName)
        with open(finished_txt, "w") as f:
            f.write("")
    
    return futures


def helperRun(command):
    # subprocess.run(command, check=True, shell=True)
    subprocess.run(command, check=True, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

### 
def specificRunnerDictSetup(args):
    """Input: args
    Output: runnerArgs dict"""
    runnerArgs = {
        "command": "eecbs",
        "args": { # These are used for detection as well
            "suboptimality": args.suboptimality,
            "cutoffTime": args.cutoffTime,
        }
    }
    return runnerArgs


def eecbs_runner_setup(args):
    global firstIter, eecbsPath
    firstIter = args.firstIter
    eecbsPath = args.eecbsPath
    assert(eecbsPath.startswith(".") and eecbsPath.endswith("eecbs") and os.path.exists(eecbsPath))


def runDataManipulator(args, ct: CustomTimer, mapsToScens, static_dict, 
                       mapsInputFolder, scensInputFolder, num_workers):
    ### Run data manipulator with multiprocessing
    ct.start("Data Manipulator")
    input_list = []
    numWorkersParallelForDataManipulator = 1
    for mapFile in mapsToScens.keys(): # mapFile is just the map name without the path or .map extension
        mapOutputFolder = static_dict[mapFile]["outputFolder"]

        mapOutputNpz = f"{args.constantMapAndBDFolder}/all_maps.npz"
        goalsOutputNpz = f"{args.constantMapAndBDFolder}/{mapFile}_goals.npz"
        bdOutputNpz = f"{args.constantMapAndBDFolder}/{mapFile}_bds.npz"

        command = " ".join(["python", "-m", "data_collection.data_manipulator",
                        f"--bdIn={mapOutputFolder}/bd", f"--goalsOutFile={goalsOutputNpz}", f"--bdOutFile={bdOutputNpz}", 
                        f"--scenIn={scensInputFolder}", f"--mapIn={mapsInputFolder}", f"--mapOutFile={mapOutputNpz}",
                        f"--num_parallel={numWorkersParallelForDataManipulator}"])
        
        input_list.append((command,))
    
    if len(input_list) > 0:
        with ray.util.multiprocessing.Pool(processes=min(len(input_list), num_workers//numWorkersParallelForDataManipulator)) as pool:
            pool.starmap(helperRun, input_list)
    else:
        print("No data manipulator runs as all paths npz files already exist")
    ct.stop("Data Manipulator")
    print("------------ Finished Data Manipulator in :{:.3f} seconds".format(ct.getTimes("Data Manipulator")))


def generic_batch_runner(args):
    """
    Generic batch runner for EECBS and Python ML model
    """
    ### Start filesystem logic
    outputFolder = args.outputFolder
    os.makedirs(outputFolder, exist_ok=True)
    scenInputFolder = args.scenFolder
    mapsInputFolder = args.mapFolder

    eecbs_runner_setup(args)

    def idToWorkerOutputFilepath(worker_id, mapName):
        # assert(worker_id >= -3)
        if worker_id == -3: # worker_id = -3 denotes glob search string for worker csvs
            return f"{outputFolder}/{mapName}/csvs/worker_*.csv"
        if worker_id == -2: # worker_id = -2 denotes the finished txt
            return f"{outputFolder}/{mapName}/finished.txt"
        if worker_id == -1: # worker_id = -1 denotes combined csv
            return f"{outputFolder}/{mapName}/csvs/combined.csv"
        return f"{outputFolder}/{mapName}/csvs/worker_{worker_id}.csv"
    
    ### Start multiprocessing logic
    ct = CustomTimer()
    ct.start("MAPF Calls")
    
    tasks = []
    num_workers = args.num_parallel_runs
    ray.init(num_cpus=num_workers)

    ### Collect scens for each map
    mapsToScens = defaultdict(list)
    # mapsToScens = dict()
    all_scen_files = list(os.listdir(scenInputFolder))
    if args.chosen_map:
        all_scen_files = [scen_name for scen_name in all_scen_files if args.chosen_map in scen_name]
    for dir_path in all_scen_files:
        if dir_path == ".DS_Store" or not dir_path.endswith(".scen"):
            continue 
        mapFile = dir_path.split("-random")[0]
        # if mapFile not in mapsToScens:
        mapsToScens[mapFile].append(scenInputFolder+"/"+dir_path)

    ### For each map, get the static information
    # This includes the map file, the scens, and the number of agents to run
    static_dict = dict()
    nameToNumRun = SharedDict.remote()
    maps_to_run = []
    for mapFile in mapsToScens: # mapFile is just the map name without the path or .map extension
        ### Create the static_dict for the map
        static_dict[mapFile] = {
            "map": f"{mapsInputFolder}/{mapFile}.map",
            "scens": mapsToScens[mapFile],
            "agentsPerScen": [], # This is used for when each scen has a specific number of agents already specified, e.g. for encountered scens
            "agentRange": [], # This is used for when we want to run a range of agents, e.g. for benchmark scens
            "outputFolder": f"{outputFolder}/{mapFile}", # This is where the output of the runs
        }
        ### Clean / remove the folders if the command is "clean"
        if args.command == "clean":
            mapOutputFolder = static_dict[mapFile]["outputFolder"]
            if os.path.exists(f"{mapOutputFolder}/bd/"):
                shutil.rmtree(f"{mapOutputFolder}/bd/") # Delete the bd folder
            if os.path.exists(f"{mapOutputFolder}/paths/"):
                if args.keepNpys:
                    # If keepNpys is true, then iterate through the paths folder and remove all files that are not npy
                    for file in os.listdir(f"{mapOutputFolder}/paths/"):
                        if not file.endswith(".npy"):
                            os.remove(f"{mapOutputFolder}/paths/{file}")
                else: # Remove the entire paths folder
                    shutil.rmtree(f"{mapOutputFolder}/paths/") # Delete the paths folder
            # Keep the csvs folder, it should just contain the combined.csv file
            continue

        ## Checks if the finished txt already exists, if so skip
        if os.path.exists(idToWorkerOutputFilepath(-2, mapFile)): # Check for finished.txt which signifies if eecbs has already been run
            print("Skipping {} as finished.txt already exists".format(mapFile))
            continue
        else: # Else, restart from scratch, so remove existing folders
            if os.path.exists(f"{outputFolder}/{mapFile}"):
                print("Removing previous potentially partial run for {}".format(mapFile))
                shutil.rmtree(f"{outputFolder}/{mapFile}", ignore_errors=False)
        maps_to_run.append(mapFile)
        
        
        agent_json_dict = None
        ### Get the number of agents to run for each scen
        if "benchmark" in scenInputFolder: # pre-loop run
             # collecting bds
            if mapFile in mapsToMaxNumAgents:
                agentNumbers = [mapsToMaxNumAgents[mapFile]]
            else:
                agentNumbers = [mapsToMaxNumAgents[mapFile.replace("-", "_")]]
            static_dict[mapFile]["agentRange"] = agentNumbers
            static_dict[mapFile]["agentsPerScen"] = [agentNumbers[0]] * len(static_dict[mapFile]["scens"])
        else: # we are somewhere in the training loop
            # scenlist = os.listdir(scenInputFolder)
            for scen in all_scen_files:
                if mapFile not in scen or not scen.endswith(".scen"):
                    continue
                # open the file
                with open(f'{scenInputFolder}/{scen}', 'r') as fh: # TODO get the path to scene file right
                    firstLine = fh.readline()
                    assert(firstLine.startswith("version "))
                    num = firstLine.split(" ")[1]
                    # agentNumbers.append(int(num))
                    static_dict[mapFile]["agentsPerScen"].append(int(num))
        
        assert(len(static_dict[mapFile]["agentsPerScen"]) > 0)
        nameToNumRun.set.remote(mapFile, len(mapsToScens[mapFile]))
        print(f"Map: {mapFile} requires {len(mapsToScens[mapFile])} runs")

    if args.command == "clean":
        return

    # Get the specific individual eecbs or pymodel arguments
    specific_runner_args = specificRunnerDictSetup(args)

    ## Create initial jobs
    for mapFile in maps_to_run:
        mapOutputFolder = static_dict[mapFile]["outputFolder"]
        os.makedirs(mapOutputFolder, exist_ok=True)
        os.makedirs(f"{mapOutputFolder}/bd/", exist_ok=True)
        # os.makedirs(f"{mapOutputFolder}/paths/", exist_ok=True)
        # os.makedirs(f"{mapOutputFolder}/csvs/", exist_ok=True)
        # pdb.set_trace()
        for scen, agentNum in zip(static_dict[mapFile]["scens"], static_dict[mapFile]["agentsPerScen"]):
            tasks.append((specific_runner_args, mapFile, agentNum, scen))

    futures = [runSingleInstanceMT.remote(nameToNumRun, num_workers, idToWorkerOutputFilepath, 
                                          static_dict, *args) for args in tasks]

    # Wait for all tasks to be processed
    while len(futures):
        ready, futures = ray.wait(futures)
        for finished_task in ready:
            children = ray.get(finished_task)
            futures += children # append child futures

    # Delete CSV files with {mapName}_worker naming convention
    for mapName in maps_to_run:
        filenames = glob.glob(idToWorkerOutputFilepath(-3, mapName))
        for filename in filenames:
            os.remove(filename)
    
    ct.stop("MAPF Calls")
    print("------------ Finished {} Calls in :{:.3f} seconds".format(args.command, ct.getTimes("MAPF Calls")))

    # Run data manipulator if running eecbs
    if args.command == "eecbs":
        runDataManipulator(args, ct, mapsToScens, static_dict, 
                           mapsInputFolder, scenInputFolder, num_workers)
    
    if args.deleteTextFiles:
        shutil.rmtree(args.outputFolder)

## Example calls of constants_generator
"""
Collecting initial bd and map data:
python -m data_collection.constants_generator --mapFolder=data_collection/data/mini_benchmark_data/maps \
                 --scenFolder=data_collection/data/mini_benchmark_data/scens \
                 --constantMapAndBDFolder=data_collection/data/benchmark_data/constant_npzs \
                 --outputFolder=data_collection/data/logs/EXP_Collect_BD/ \
                 --num_parallel_runs=50 \
                 --deleteTextFiles=true \
                 "eecbs" \
                 --firstIter=true --cutoffTime=1
                 
python -m data_collection.constants_generator --mapFolder=data_collection/data/benchmark_data/maps \
                 --scenFolder=data_collection/data/benchmark_data/scens \
                 --constantMapAndBDFolder=data_collection/data/benchmark_data/constant_npzs2 \
                 --outputFolder=data_collection/data/logs/EXP_Collect_BD2/ \
                 --num_parallel_runs=50 \
                 --deleteTextFiles=true \
                 "eecbs" \
                 --firstIter=true --cutoffTime=1

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument("--mapFolder", help="contains all scens to run", type=str, required=True)
    parser.add_argument("--scenFolder", help="contains all scens to run", type=str, required=True)
    # numAgentsHelp = "Number of agents per scen; [int1,int2,..] or `increment` for all agents up to the max"
    # parser.add_argument("--numAgents", help=numAgentsHelp, type=str, required=True)
    parser.add_argument("--constantMapAndBDFolder", help="output folder for precomputed map and bds", type=str, required=True)
    parser.add_argument("--outputFolder", help="parent output folder where each map folder will contain paths/ and csvs/ results", 
                        type=str, required=True)
    parser.add_argument('--num_parallel_runs', help="How many multiple maps in parallel tmux sessions. 1 = No parallel runs.", 
                        type=int, required=True)
    parser.add_argument('--chosen_map', help="For benchmarking choose just one map from all the maps", 
                        type=str, default=None)
    parser.add_argument('--deleteTextFiles', help="delete outputFolder when done", 
                        type=bool, default=False)
    # parser.add_argument("--iter", help="iteration number", type=int, default=0) #this is used only for the seed for simulation now

    # Subparses for C++ EECBS or Python ML model
    subparsers = parser.add_subparsers(dest="command", required=True)

    ### Clean parser, no additional arguments are needed
    clean_parser = subparsers.add_parser("clean", help="Clean up the output folders")
    clean_parser.add_argument("--keepNpys", help="Keep the npy files for pymodel outputs", type=lambda x: bool(str2bool(x)), required=True)

    ### EECBS parser
    eecbs_parser = subparsers.add_parser("eecbs", help="Run eecbs")
    eecbs_parser.add_argument("--eecbsPath", help="path to eecbs executable", type=str, default="./data_collection/eecbs/build_release3/eecbs")
    eecbs_parser.add_argument('--firstIter', dest='firstIter', type=lambda x: bool(str2bool(x)))
    # EECBS driver parameters
    eecbs_parser.add_argument("--cutoffTime", help="cutoffTime", type=int, default=60)
    eecbs_parser.add_argument("--suboptimality", help="suboptimality", type=float, default=2)
    # eecbs_parser.add_argument("--expnum", help="experiment number", type=int, default=0)
    
    ### Parse arguments and run the batch runner
    args = parser.parse_args()
    generic_batch_runner(args)