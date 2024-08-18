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


import multiprocessing
# from custom_utils.custom_timer import CustomTimer
from custom_utils.custom_timer import CustomTimer
from custom_utils.common_helper import str2bool, getMapBDScenAgents
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

mapsToMaxNumAgents = {
    "Berlin_1_256": 1000,
    "Boston_0_256": 1000,
    "Paris_1_256": 1000,
    "brc202d": 1000,
    "den312d": 1000, 
    "den520d": 1000,
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
    "random_32_32_20": 409,
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
    mapname, bdname, scenname, _ = getMapBDScenAgents(scenfile)
    bd_path = f"{outputFolder}/bd/{bdname}.{numAgents}.txt"
    # scenname = (scenfile.split("/")[-1])
    # mapname = mapfile.split("/")[-1].split(".")[0]
    # bd_path = f"{outputFolder}/bd/{scenname}{numAgents}.txt"
    # command = f".{file_home}/eecbs/build_release2/eecbs"
    command = f"{eecbsPath}"

    for aKey in eecbsArgs["args"]:
        command += " --{}={}".format(aKey, eecbsArgs["args"][aKey])
    # tempOutPath = f"{outputFolder}/paths/{scenname}{numAgents}.txt"
    outputPathFile = f"{outputFolder}/paths/{bdname}.{scenname}.{numAgents}.txt"
    command += " --agentNum={} --agents={} --outputPaths={} --firstIter={} --bd_file={}".format(
                numAgents, scenfile, outputPathFile, firstIter, bd_path)
    # command += " --agentNum={} --seed={} --agentsFile={}".format(numAgents, seed, scenfile)
    command += " --output={} --map={}".format(outputfile, mapfile)
    # print(command.split("suboptimality=1")[1])
    return command
    
def getPyModelCommand(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile):
    """Command for running Python model"""
    # scenname = (scenfile.split("/")[-1])
    # mapname = mapfile.split("/")[-1].split(".")[0]
    mapname, bdname, scenname, _ = getMapBDScenAgents(scenfile)
    command = ""
    if runnerArgs["condaEnv"] is not None:
        command += "conda activate {} && ".format(runnerArgs["condaEnv"]) # e.g. conda activate pytorchfun && python -m gnn.simulator2
    command += "python -m gnn.simulator2"

    # Simulator parameters
    for aKey in runnerArgs["args"]:
        command += " --{}={}".format(aKey, runnerArgs["args"][aKey])
    
    command += f" --mapNpzFile=data_collection/data/benchmark_data/constant_npzs/all_maps.npz"
    command += f" --mapName={mapname} --scenFile={scenfile} --agentNum={numAgents}"
    bdFile = f"data_collection/data/benchmark_data/constant_npzs/{mapname}_bds.npz"
    command += f" --bdNpzFile={bdFile}"
    command += f" --outputCSVFile={outputfile}"
    # tempOutPath = f"{outputFolder}/paths/{scenname}{numAgents}.npy" # Note scenname ends with a .scen
    outputPathNpy = f"{outputFolder}/paths/{bdname}.{scenname}.npy"
    command += f" --outputPathsFile={outputPathNpy}"
    command += f" --numScensToCreate={runnerArgs['numScensToCreate']}"
    command += f" --percentSuccessGenerationReduction={runnerArgs['percentSuccessGenerationReduction']}"
    return command

def getCommandForSingleInstance(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile):
    if runnerArgs["command"] == "eecbs":
        return getEECBSCommand(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile)
    elif runnerArgs["command"] == "pymodel":
        return getPyModelCommand(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile)
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
        if aKey == "extra_layers" or aKey == "bd_pred":
            continue
        if aKey not in df.columns:
            
            raise KeyError("Error: {} not in the columns of the dataframe".format(aKey))
        df = df[df[aKey] == aValue]  # Filter the dataframe to only include the runs with the same parameters
    
    # EECBS and pymodel have different commands for inputting map, agents, and agentNum
    if runnerArgs["command"] == "eecbs":
        df = df[(df["map"] == mapfile) & (df["agents"] == scenfile) & (df["agentNum"] == aNum)]
    elif runnerArgs["command"] == "pymodel":
        pymodel_map_name = mapfile.split("/")[-1].removesuffix(".map")
        assert(pymodel_map_name in mapsToMaxNumAgents.keys())
        df = df[(df["mapName"] == pymodel_map_name) & (df["scenFile"] == scenfile) & (df["agentNum"] == aNum)]
    else:
        raise KeyError("Unknown command: {}".format(runnerArgs["command"]))
    
    ### Checks if the corresponding runs in the df have been completed already
    if len(df) > 0:
        # assert(len(df) == 1)
        if len(df) > 1:
            print("Warning, multiple runs with the same parameters, likely due to a previous crash")
            print("Map: {}, NumAgents: {}, Scen: {}, # Found: {}".format(mapfile, aNum, scenfile, len(df)))
        if runnerArgs["command"] == "eecbs":
            success = df["solution cost"].values[-1] != -1
        elif runnerArgs["command"] == "pymodel":
            success = df["success"].values[0] == 1
        else:
            raise KeyError("Unknown command: {}".format(runnerArgs["command"]))
        # success = df["overall_solution"].values[0] == 1
        return True, success
    else:
        return False, 0


####### Multi test
def runSingleInstanceMT(queue, nameToNumRun, lock, worker_id, idToWorkerOutputFilepath, static_dict, 
                        runnerArgs, mapName, curAgentNum, scen):
    workerOutputCSV = idToWorkerOutputFilepath(worker_id, mapName)
    combined_filename = idToWorkerOutputFilepath(-1, mapName) # -1 denotes combined
    mapFile = static_dict[mapName]["map"]
    outputFolder = static_dict[mapName]["outputFolder"]

    runBefore, status = detectExistingStatus(runnerArgs, mapFile, curAgentNum, scen, combined_filename) # check in combinedDf
    if not runBefore:
        command = getCommandForSingleInstance(runnerArgs, outputFolder, workerOutputCSV, mapFile, curAgentNum, scen)
        # print(command)
        runCommandWithTmux(worker_id, command)
        runBefore, status = detectExistingStatus(runnerArgs, mapFile, curAgentNum, scen, workerOutputCSV)  # check in worker's df
        if not runBefore:
            # print(f"worker_id:{worker_id}")
            print(f"mapFile:{mapFile}, curAgentNum:{curAgentNum}, scen:{scen}, workerOutputCSV:{workerOutputCSV}, worker_id:{worker_id}")
            raise RuntimeError("Fix detectExistingStatus; we ran an instance but cannot find it afterwards!")

    ## Update the number of runs. If we are the last one, then check if we should run the next agent number.
    lock.acquire()
    nameToNumRun[mapName] -= 1
    assert(nameToNumRun[mapName] >= 0)
    if nameToNumRun[mapName] == 0:
        queue.put(("checkIfRunNextAgents", (runnerArgs, mapName, curAgentNum)))
    lock.release()


def checkIfRunNextAgents(queue, nameToNumRun, lock, num_workers, idToWorkerOutputFilepath, static_dict, eecbsArgs, mapName, curAgentNum):
    """Inputs:
        curAgentNum: This is only used if static_dict["agentRange"] is not empty
    """
    ## Load separate CSVs and combine them
    combined_dfs = []
    for i in range(num_workers):
        filename = idToWorkerOutputFilepath(i, mapName)
        if os.path.exists(filename):
            combined_dfs.append(pd.read_csv(filename, index_col=False))
    
    combined_filename = idToWorkerOutputFilepath(-1, mapName) # -1 denotes combined
    if os.path.exists(combined_filename):
        combined_dfs.append(pd.read_csv(combined_filename, index_col=False))
    combined_df = pd.concat(combined_dfs)
    combined_df = combined_df.drop_duplicates()
    if len(static_dict[mapName]["agentRange"]) > 0:
        print("Completed map: {}, agentNum: {}, combined_df: {}".format(mapName, curAgentNum, combined_df.shape))
    else:
        unique_agents = set(static_dict[mapName]["agentsPerScen"])
        print("Completed map: {}, agentNums: {}, combined_df: {}".format(mapName, unique_agents, combined_df.shape))
    combined_df.to_csv(combined_filename, index=False)
    
    ## Check status on current runs
    mapFile = static_dict[mapName]["map"]
    numSuccess = 0
    numToRunTotal = len(static_dict[mapName]["scens"])
    for scen, agentNum in zip(static_dict[mapName]["scens"], static_dict[mapName]["agentsPerScen"]):
        runBefore, status = detectExistingStatus(eecbsArgs, mapFile, agentNum, scen, combined_df)
        if not runBefore:
            print("Error: {}, {}, {}, {}".format(mapFile, agentNum, scen, combined_filename))
        assert(runBefore)
        numSuccess += status

    # shouldRunDataManipulator = False
    finished = False
    if len(static_dict[mapName]["agentRange"]) > 0: # If we are running a range of agents, check the next one
        print(numSuccess, numToRunTotal, curAgentNum, mapName)
        if numSuccess < numToRunTotal/2:
            print("Early terminating as only succeeded {}/{} for {} agents on map {}".format(
                                            numSuccess, numToRunTotal, curAgentNum, mapName))
            # shouldRunDataManipulator = True
            finished = True
        else:
            scens = static_dict[mapName]["scens"]
            agentNumsToRun = static_dict[mapName]["agentRange"]
            assert(curAgentNum in agentNumsToRun)
            if agentNumsToRun.index(curAgentNum) + 1 == len(agentNumsToRun):
                print("Finished all agent numbers for map: {}".format(mapName))
                # shouldRunDataManipulator = True
                finished = True
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
        finished = True
        # shouldRunDataManipulator = True

    if finished:
        finished_txt = idToWorkerOutputFilepath(-2, mapName)
        with open(finished_txt, "w") as f:
            f.write("")
    
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
            worker_id: int, num_workers: int, static_dict, idToWorkerOutputFilepath):
    while True:
        task = queue.get()
        if task is None:
            break
        func, args = task
        if func == "checkIfRunNextAgents":
            checkIfRunNextAgents(queue, nameToNumRun, lock, num_workers, idToWorkerOutputFilepath, static_dict, *args)
        elif func == "runSingleInstanceMT":
            runSingleInstanceMT(queue, nameToNumRun, lock, worker_id, idToWorkerOutputFilepath, static_dict, *args)
        # elif func == "runDataManipulator":
        #     runDataManipulator(worker_id, *args)
        else:
            raise ValueError("Unknown function")
        queue.task_done()

def helperRun(command):
    subprocess.run(command, check=True, shell=True)

### 
def specificRunnerDictSetup(args):
    """Input: args
    Output: runnerArgs dict, differs depending on eecbs or pymodel"""
    if args.command == "eecbs":
        runnerArgs = {
            "command": "eecbs",
            "args": { # These are used for detection as well
                "suboptimality": args.suboptimality,
                "cutoffTime": args.cutoffTime,
            }
        }
    elif args.command == "pymodel":
        runnerArgs = {
            "command": "pymodel",
            "condaEnv": args.condaEnv,
            "args": { # These are used for detection as well
                "modelPath": args.modelPath,
                "useGPU": args.useGPU,
                "k": args.k,
                "m": args.m,
                "maxSteps": args.maxSteps,
                "shieldType": args.shieldType,
                "lacamLookahead": args.lacamLookahead,
            },
            "percentSuccessGenerationReduction": args.percentSuccessGenerationReduction,
            "numScensToCreate": args.numScensToCreate
        }
        if args.extra_layers is not None:
            runnerArgs["args"]["extra_layers"]=args.extra_layers
        if args.bd_pred is not None:
            runnerArgs["args"]["bd_pred"]=args.bd_pred
    else:
        raise ValueError("Unknown command: {}".format(args.command))
    return runnerArgs

def eecbs_runner_setup(args):
    global firstIter, eecbsPath
    firstIter = args.firstIter
    eecbsPath = args.eecbsPath
    assert(eecbsPath.startswith(".") and eecbsPath.endswith("eecbs") and os.path.exists(eecbsPath))

    # Create folder for saving data_manipulator npz files
    # Do this now so we can check if the folder exists before running the eecbs
    outputPathNpzFolder = args.outputPathNpzFolder
    if outputPathNpzFolder is None:
        outputPathNpzFolder = f"{args.outputFolder}/../eecbs_npzs"
    os.makedirs(outputPathNpzFolder, exist_ok=True)

def runDataManipulator(args, ct: CustomTimer, mapsToScens, static_dict, 
                       outputPathNpzFolder, mapsInputFolder, num_workers):
    ### Run data manipulator with multiprocessing
    ct.start("Data Manipulator")
    input_list = []
    numWorkersParallelForDataManipulator = 1
    for mapFile in mapsToScens.keys(): # mapFile is just the map name without the path or .map extension
        mapOutputFolder = static_dict[mapFile]["outputFolder"]
        pathsIn = f"{mapOutputFolder}/paths/"

        # mapOutputNpz = f".{file_home}/data/benchmark_data/constant_npzs/{mapFile}_map.npz"
        # bdOutputNpz = f".{file_home}/data/benchmark_data/constant_npzs/{mapFile}_bds.npz"
        # mapOutputNpz = f"{args.constantMapAndBDFolder}/{mapFile}_map.npz"
        mapOutputNpz = f"{args.constantMapAndBDFolder}/all_maps.npz" #TODO change this to previous line for testing new maps
        bdOutputNpz = f"{args.constantMapAndBDFolder}/{mapFile}_bds.npz"
        # mapOutputNpz = f"data_collection/data/benchmark_data/constant_npz2/{mapFile}_map.npz"
        # bdOutputNpz = f"data_collection/data/benchmark_data/constant_npzs2/{mapFile}_bds.npz"
        pathOutputNpz = f"{outputPathNpzFolder}/{mapFile}_paths.npz"
        if os.path.exists(pathOutputNpz):
            print("Skipping: {} as paths npz already exists".format(mapFile))
            continue

        command = " ".join(["python", "-m", "data_collection.data_manipulator", 
                        f"--pathsIn={pathsIn}", f"--pathOutFile={pathOutputNpz}",
                        f"--bdIn={mapOutputFolder}/bd", f"--bdOutFile={bdOutputNpz}", 
                        f"--mapIn={mapsInputFolder}", f"--mapOutFile={mapOutputNpz}",
                        f"--num_parallel={numWorkersParallelForDataManipulator}"])
        input_list.append((command,))
        # pdb.set_trace()
    
    if len(input_list) > 0:
        with multiprocessing.Pool(processes=min(len(input_list), num_workers//numWorkersParallelForDataManipulator)) as pool:
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

    if args.command == "eecbs":
        eecbs_runner_setup(args)

    def idToWorkerOutputFilepath(worker_id, mapName):
        assert(worker_id >= -2)
        if worker_id == -2: # worker_id = -2 denotes the finished txt
            return f"{outputFolder}/{mapName}/finished.txt"
        if worker_id == -1: # worker_id = -1 denotes combined csv
            return f"{outputFolder}/{mapName}/csvs/combined.csv"
        return f"{outputFolder}/{mapName}/csvs/worker_{worker_id}.csv"
    
    ### Start multiprocessing logic
    ct = CustomTimer()
    ct.start("MAPF Calls")
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    queue = multiprocessing.JoinableQueue()
    num_workers = args.num_parallel_runs
    workers = []

    ### Collect scens for each map
    mapsToScens = defaultdict(list)
    # mapsToScens = dict()
    all_scen_files = list(os.listdir(scenInputFolder))
    for dir_path in all_scen_files:
        if dir_path == ".DS_Store" or not dir_path.endswith(".scen"):
            continue 
        mapFile = dir_path.split("-")[0]
        # if mapFile not in mapsToScens:
        mapsToScens[mapFile].append(scenInputFolder+"/"+dir_path)

    ### For each map, get the static information
    # This includes the map file, the scens, and the number of agents to run
    static_dict = dict()
    nameToNumRun = manager.dict()
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

        ### Checks if the finished txt already exists, if so skip
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
            if args.numAgents == "increment":
                increment = min(100,  mapsToMaxNumAgents[mapFile])
                maximumAgents = mapsToMaxNumAgents[mapFile] + 1
                agentNumbers = list(range(increment, maximumAgents, increment))
            elif ".json" in args.numAgents:
                if agent_json_dict is None:
                    f = open(args.numAgents)
                    agent_json_dict = json.load(f)['map_agent_counts']
                agentNumbers = agent_json_dict[mapFile]
            else:
                agentNumbers = [int(x) for x in args.numAgents.split(",")]
            # maximumAgents = increment + 1 # Just run one setting as of now
            # agentNumbers = [100, 200, 300][:1]
            # agentNumbers = [10, 50, 100]
            # agentNumbers = [mapsToMaxNumAgents[mapFile]] # Only do this to collect bds
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
        nameToNumRun[mapFile] = len(mapsToScens[mapFile])
        # print(f"Map: {mapFile} requires {nameToNumRun[mapFile]} runs")

    if args.command == "clean":
        return

    # Start worker processes with corresponding tmux session
    for worker_id in range(num_workers):
        createTmuxSession(worker_id)
        p = multiprocessing.Process(target=worker, 
                    args=(queue, nameToNumRun, lock, worker_id, num_workers, static_dict, idToWorkerOutputFilepath))
        p.start()
        workers.append(p)

    # Get the specific individual eecbs or pymodel arguments
    specific_runner_args = specificRunnerDictSetup(args)

    ## Create initial jobs
    for mapFile in maps_to_run:
        mapOutputFolder = static_dict[mapFile]["outputFolder"]
        os.makedirs(mapOutputFolder, exist_ok=True)
        os.makedirs(f"{mapOutputFolder}/bd/", exist_ok=True)
        os.makedirs(f"{mapOutputFolder}/paths/", exist_ok=True)
        os.makedirs(f"{mapOutputFolder}/csvs/", exist_ok=True)
        # pdb.set_trace()
        for scen, agentNum in zip(static_dict[mapFile]["scens"], static_dict[mapFile]["agentsPerScen"]):
            queue.put(("runSingleInstanceMT", (specific_runner_args, mapFile, agentNum, scen)))

    # Wait for all tasks to be processed
    queue.join()

    # Stop worker processes
    for i in range(num_workers):
        killTmuxSession(i)
        queue.put(None)
    for p in workers:
        p.join()

    # Delete CSV files with {mapName}_worker naming convention
    for mapName in maps_to_run:
        for worker_id in range(num_workers):
            filename = idToWorkerOutputFilepath(worker_id, mapName)
            if os.path.exists(filename):
                os.remove(filename)
    
    ct.stop("MAPF Calls")
    print("------------ Finished {} Calls in :{:.3f} seconds".format(args.command, ct.getTimes("MAPF Calls")))

    # ### Clean up the bds, csvs, and paths folders
    # for mapFile in mapsToScens:
    #     mapOutputFolder = static_dict[mapFile]["outputFolder"]
    #     if os.path.exists(f"{mapOutputFolder}/bd/"):
    #         shutil.rmtree(f"{mapOutputFolder}/bd/") # Delete the bd folder
    #     if os.path.exists(f"{mapOutputFolder}/paths/"):
    #         shutil.rmtree(f"{mapOutputFolder}/paths/") # Delete the paths folder
    #     # Keep the csvs folder, it should just contain the combined.csv file
    
    # Run data manipulator if running eecbs
    if args.command == "eecbs":
        runDataManipulator(args, ct, mapsToScens, static_dict, 
                           args.outputPathNpzFolder, mapsInputFolder, num_workers)


## Example calls of BatchRunner3
"""
Note: These are likely outdated, but the general structure should be the same
python -m data_collection.eecbs_batchrunner3 --mapFolder=data_collection/data/benchmark_data/maps \
                 --scenFolder=data_collection/data/benchmark_data/scens \
                 --constantMapAndBDFolder=data_collection/data/benchmark_data/constant_npzs2 \
                 --outputFolder=data_collection/data/logs/EXP_Test_batch/iter0/[eecbs_outputs or pymodel_outputs] \
                 --num_parallel_runs=50 \
EECBS specific:
                 "eecbs" \
                 --outputPathNpzFolder=data_collection/data/logs/EXP_Test_batch/iter0/eecbs_npzs \
                 --firstIter=false --cutoffTime=5 \
Python model specific:
                 "pymodel" \
                 --modelPath=data_collection/data/logs/EXP_Test2/iter0/models/max_test_acc.pt \
                 --k=4 --m=5 --maxSteps=100 --shieldType=CS-PIBT \
Cleaning the output folders (to save memory):
python -m data_collection.eecbs_batchrunner3 --mapFolder=data_collection/data/benchmark_data/maps \
                 --scenFolder=data_collection/data/benchmark_data/scens \
                 --constantMapAndBDFolder=data_collection/data/benchmark_data/constant_npzs2 \
                 --outputFolder=data_collection/data/logs/EXP_Test_batch/iter0/[eecbs_outputs or pymodel_outputs] \
                 --numAgents=1 \ 
                 --num_parallel_runs=1 \
                 "clean" --keepNpys=True

Collecting initial bd and map data:
python -m data_collection.eecbs_batchrunner3 --mapFolder=data_collection/data/benchmark_data/maps \
                 --scenFolder=data_collection/data/benchmark_data/scens \
                 --constantMapAndBDFolder=data_collection/data/benchmark_data/constant_npzs2 \
                 --outputFolder=data_collection/data/logs/EXP_Collect_BD/iter0/eecbs_outputs \
                 --num_parallel_runs=50 \
                 "eecbs" \
                 --outputPathNpzFolder=data_collection/data/logs/EXP_Collect_BD/iter0/eecbs_npzs \
                 --firstIter=true --cutoffTime=1
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument("--mapFolder", help="contains all scens to run", type=str, required=True)
    parser.add_argument("--scenFolder", help="contains all scens to run", type=str, required=True)
    numAgentsHelp = "Number of agents per scen; [int1,int2,..] or `increment` for all agents up to the max"
    parser.add_argument("--numAgents", help=numAgentsHelp, type=str, required=True)
    parser.add_argument("--constantMapAndBDFolder", help="contains the precomputed map and bd folders", type=str, required=True)
    parser.add_argument("--outputFolder", help="parent output folder where each map folder will contain paths/ and csvs/ results", 
                        type=str, required=True)
    parser.add_argument('--num_parallel_runs', help="How many multiple maps in parallel tmux sessions. 1 = No parallel runs.", 
                        type=int, required=True)

    # Subparses for C++ EECBS or Python ML model
    subparsers = parser.add_subparsers(dest="command", required=True)

    ### Clean parser, no additional arguments are needed
    clean_parser = subparsers.add_parser("clean", help="Clean up the output folders")
    clean_parser.add_argument("--keepNpys", help="Keep the npy files for pymodel outputs", type=lambda x: bool(str2bool(x)), required=True)

    ### EECBS parser
    eecbs_parser = subparsers.add_parser("eecbs", help="Run eecbs")
    eecbs_parser.add_argument("--outputPathNpzFolder", help="folder for data_manipulator to create npz files", type=str, default=None)
    eecbs_parser.add_argument("--eecbsPath", help="path to eecbs executable", type=str, default="./data_collection/eecbs/build_release2/eecbs")
    eecbs_parser.add_argument('--firstIter', dest='firstIter', type=lambda x: bool(str2bool(x)))
    # EECBS driver parameters
    eecbs_parser.add_argument("--cutoffTime", help="cutoffTime", type=int, default=60)
    eecbs_parser.add_argument("--suboptimality", help="suboptimality", type=float, default=2)
    # eecbs_parser.add_argument("--expnum", help="experiment number", type=int, default=0)
    # eecbs_parser.add_argument("--iter", help="iteration number", type=int)

    ### Python ML Model
    pymodel_parser = subparsers.add_parser("pymodel", help="Run python model")
    pymodel_parser.add_argument('--condaEnv', type=str, help="name of conda env to activate for simulator2.py", required=False)
    # Simulator parameters
    pymodel_parser.add_argument('--modelPath', type=str, required=True)
    pymodel_parser.add_argument('--useGPU', type=lambda x: bool(str2bool(x)), required=True)
    pymodel_parser.add_argument('--k', type=int, help="local window size", required=True)
    pymodel_parser.add_argument('--m', type=int, help="number of closest neighbors", required=True)
    pymodel_parser.add_argument('--maxSteps', type=str, help="see simulator2", required=True)
    pymodel_parser.add_argument('--shieldType', type=str, default='CS-PIBT')
    extraLayersHelp = "Types of additional layers for training, comma separated. Options are: agent_locations, agent_goal, at_goal_grid"
    pymodel_parser.add_argument('--extra_layers', help=extraLayersHelp, type=str, default=None)
    pymodel_parser.add_argument('--bd_pred', type=str, default=None, help="bd_predictions added to NN, type anything if adding")
    pymodel_parser.add_argument('--lacamLookahead', type=int, default=0)
    # Output parameters
    pymodel_parser.add_argument('--numScensToCreate', help="see simulator2", type=int, required=True)
    pymodel_parser.add_argument('--percentSuccessGenerationReduction', help="see simulator2", type=float, required=True)
    
 

    ### Parse arguments and run the batch runner
    args = parser.parse_args()
    generic_batch_runner(args)