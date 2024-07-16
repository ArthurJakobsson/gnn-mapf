import os
import argparse
import subprocess  # For executing eecbs script
import pandas as pd  # For smart batch running
import pdb # For debugging
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For utils
from collections import defaultdict
import shutil

####### Set the font size of the plots
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# mapsToMaxNumAgents = {
#     "Berlin_1_256": 100, # TODO: unverified
#     "Boston_0_256": 100, # TODO: unverified
#     "Paris_1_256": 1000, # Verified
#     "random_32_32_20": 409, # Verified
#     "random_32_32_10": 461, # Verified
#     "den520d": 1000, # Verified
#     "den312d": 1000, # Verified
#     "empty_32_32": 511, # Verified
#     "empty_48_48": 1000, # Verified
#     "ht_chantry": 1000, # Verified
#     "warehouse_10_20_10_2_2": 101,
#     "warehouse_20_40_10_2_2": 101
# }
mapsToMaxNumAgents = { #TODO change this to 100 for all
    "Berlin_1_256": 101,
    "lt_gallowstemplar_n": 101, 
    "final_test9": 101,
    "empty_8_8": 101,
    "den312d": 101, 
    "final_test2": 101,
    "final_test6": 101,
    "random_32_32_10": 101,
    "brc202d": 101,
    "room_64_64_8": 101,
    "maze_128_128_1": 101,
    "warehouse_20_40_10_2_2": 101,
    "final_test3": 101,
    "Paris_1_256": 101,
    "final_test8": 101,
    "maze_128_128_10": 101, 
    "w_woundedcoast": 101,
    "maze_32_32_4": 101,
    "maze_32_32_2": 101,
    "ht_chantry": 101,
    "final_test1": 101,
    "empty_48_48": 101,
    "random_64_64_20": 101,
    "room_64_64_16": 101,
    "final_test4": 101,
    "empty_32_32": 101,
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
    "random_32_32_20": 101,
    "lak303d": 101,
    "den520d": 101
}

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def runOnSingleInstance(eecbsArgs, numAgents, scenfile, scenname):
    # ### Instance
    # command = "./build_release/eecbs -m {} -a {}".format(mapfile, scenfile)
    # command += " --seed={} -k {}".format(seed, numAgents)
    # command += " -t {}".format(eecbsArgs["timeout"])
    # command += " --output={}".format(eecbsArgs["output"])
    # command += " --outputPaths={}".format(eecbsArgs["outputPaths"])

    # ### EECBS parameters
    # command += " --suboptimality={}".format(eecbsArgs["suboptimality"])

    # ### W-EECBS parameters
    # command += " --useWeightedFocalSearch={}".format(eecbsArgs["useWeightedFocalSearch"])
    # if (eecbsArgs["useWeightedFocalSearch"]):
    #     command += " --r_weight={}".format(eecbsArgs["r_weight"])
    #     command += " --h_weight={}".format(eecbsArgs["h_weight"])
    # print(os.path.abspath(os.getcwd()))
    command = f".{file_home}/eecbs/build_release/eecbs"
    for aKey in eecbsArgs:
        command += " --{}={}".format(aKey, eecbsArgs[aKey])
    tempOutPath = f".{file_home}/eecbs/raw_data/paths/{scenname}{numAgents}.txt"
    command += " --agentNum={} --agents={} --outputPaths={} --firstIter={} --scenname={}".format(numAgents, scenfile, tempOutPath, firstIter, scenname)
    print(command)
    subprocess.run(command.split(" "), check=True) # True if want failure error
    
    
def detectExistingStatus(aNum, scenname): # TODO update
    """
    Output:
        If has been run before
        Success if run before
    """
    # this is the same output path made in runOnSingleInstance

    tempOutPath = f".{file_home}/eecbs/raw_data/paths/{scenname}{aNum}.txt"

    if not os.path.exists(tempOutPath):
        return False, 0
    return True, 100
    # df = pd.read_csv(eecbsArgs["outputPaths"])

    # ### Checks if the corresponding runs in the df have been completed already
    # for aKey, aValue in eecbsArgs.items():
    #     ### If this is false, then we don't care about the r_weight and h_weight
    #     if not eecbsArgs["useWeightedFocalSearch"]:
    #         if aKey == "r_weight":
    #             continue
    #         if aKey == "h_weight":
    #             continue
    #     if aKey == "output":
    #         continue
    #     df = df[df[aKey] == aValue]  # Filter the dataframe to only include the runs with the same parameters
    # df = df[(df["agentsFile"] == scen) & (df["agentNum"] == aNum) & (df["seed"] == seed)]
    # if len(df) > 0:
    #     assert(len(df) == 1)
    #     success = (df["solution cost"] != -1).values[0]
    #     return True, success
    # else:
    #     return False, 0

def runOnSingleMap(eecbsArgs, mapName, agentNumbers, scens, inputFolder):
    print("Single Map")
    if "benchmark" in inputFolder:
        for aNum in agentNumbers:
            # print("Starting to run {} agents on map {}".format(aNum, mapName))
            numSuccess = 0
            status=0
            numToRunTotal = len(scens)
            for scen in scens:
                scenname = (scen.split("/")[-1])
                runBefore, status = detectExistingStatus(aNum, scenname)
                runBefore=False #TODO fix detectExisting
                if not runBefore:
                    print(scen)
                    runOnSingleInstance(eecbsArgs, aNum, scen, scenname)
                    runBefore, status = detectExistingStatus(aNum, scenname)
                    assert(runBefore)
                    status+=1
                numSuccess += status

            if numSuccess < numToRunTotal/2:
                print("Early terminating as only succeeded {}/{} for {} agents on map {}".format(
                                                numSuccess, numToRunTotal, aNum, mapName))
                break
    else:
        for aNum, scen in zip(agentNumbers, scens): # for each scen, run its agent number
            print("Starting to run {} agents on map {}".format(aNum, mapName))
            scenname = (scen.split("/")[-1])
            runBefore, status = detectExistingStatus(aNum, scenname)
            runBefore=False #TODO fix detectExisting
            if not runBefore:                
                runOnSingleInstance(eecbsArgs, aNum, scen, scenname)
                runBefore, status = detectExistingStatus(aNum, scenname)
                assert(runBefore)

def eecbs_runner(args):
    """
    Compare the runtime of EECBS and W-EECBS
    """
    # if args.mapName not in mapsToMaxNumAgents:
    #     raise KeyError("Map name {} not found in mapsToMaxNumAgents. Please add it to mapsToMaxNumAgents into the top of batch_script.py".format(args.mapName))

    ### Create the folder for the output file if it does not exist

    # outputCSV = " bd_stats" + ".csv"
    # totalOutputPath = f".{file_home}/eecbs/raw_data/bd/{outputCSV}"
    # if not os.path.exists(os.path.dirname(totalOutputPath)):
    #     os.makedirs(os.path.dirname(totalOutputPath))


    mapsToScens = defaultdict(list)
    for dir_path in os.listdir(scenInputFolder):
        mapFile = dir_path.split("-")[0]
        mapsToScens[mapFile].append(scenInputFolder+"/"+dir_path)

    for mapFile in mapsToScens:
        eecbsArgs = {
            "map": f"{mapsInputFolder}/{mapFile}.map",
            "output": f".{file_home}/eecbs/raw_data/bd/stats.csv",
            # "outputPaths": f".{file_home}/eecbs/raw_data/paths/paths.txt",
            "suboptimality": args.suboptimality,
            "cutoffTime": args.cutoffTime
        }
        # scens = helperCreateScens(args.num_scens, args.mapName, args.dataPath)
        scens = mapsToScens[mapFile]

        if "benchmark" in scenInputFolder: # pre-loop run
            increment = 100
            agentNumbers = list(range(increment, mapsToMaxNumAgents[mapFile]+1, increment))

            ### Run baseline EECBS
            runOnSingleMap(eecbsArgs, mapFile, agentNumbers, scens, scenInputFolder)

        else: # we are somewhere in the training loop
            agentNumbers = []
            scenlist = os.listdir(scenInputFolder)
            for scen in scenlist:
                # open the file
                with open(f'{scenInputFolder}/{scen}', 'r') as fh: # TODO get the path to scene file right
                    line = fh.readline()
                    number = line.split(" ")[1]
                    agentNumbers.append(int(number))
            # run eecbs
            runOnSingleMap(eecbsArgs, mapFile, agentNumbers, scens, scenInputFolder)

        # pdb.set_trace()
        # move the new eecbs output
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}", exist_ok=True)
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/bd/", exist_ok=True)
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/paths/", exist_ok=True)

        bd_files = os.listdir(f".{file_home}/eecbs/raw_data/bd/")
        path_files = os.listdir(f".{file_home}/eecbs/raw_data/paths/")
        for file_name in bd_files:
            if file_name.endswith(".csv"):
                continue
            shutil.move(os.path.join(f".{file_home}/eecbs/raw_data/bd/", file_name), f".{file_home}/eecbs/raw_data/{mapFile}/bd/{file_name}") 
        for file_name in path_files:
            shutil.move(os.path.join(f".{file_home}/eecbs/raw_data/paths/", file_name), f".{file_home}/eecbs/raw_data/{mapFile}/paths/{file_name}") 
        
        # shutil.move(f".{file_home}/eecbs/raw_data/bd/", f".{file_home}/eecbs/raw_data/" + mapFile)
        # shutil.move(f".{file_home}/eecbs/raw_data/paths/", f".{file_home}/eecbs/raw_data/{mapFile}")
        # os.mkdir(f".{file_home}/eecbs/raw_data/paths/")
        

        # make the npz files
        # TODO don't hardcode exp, iter numbers
        subprocess.run(["python", "./data_collection/data_manipulator.py", f"--pathsIn=.{file_home}/eecbs/raw_data/{mapFile}/paths/", 
                        f"--bdIn=.{file_home}/eecbs/raw_data/{mapFile}/bd/", 
                        f"--mapIn={mapsInputFolder}", f"--trainOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/train_{mapFile}_{args.iter}", 
                        f"--valOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/val_{mapFile}_{args.iter}"])
        

        # os.mkdir(f".{file_home}/eecbs/raw_data/bd")
        # os.mkdir(f".{file_home}/eecbs/raw_data/paths")
        shutil.rmtree(f".{file_home}/eecbs/raw_data/{mapFile}/paths") # clean path files once they have been recorded
        os.mkdir(f".{file_home}/eecbs/raw_data/{mapFile}/paths") # remake "path" folder


# python batch_runner.py den312d --logPath data/logs/test --cutoffTime 10 --suboptimality 2
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mapFolder", help="contains all scens to run", type=str)
    parser.add_argument("--scenFolder", help="contains all scens to run", type=str)
    parser.add_argument("--outputFolder", help="place to output all eecbs output", type=str)
    parser.add_argument("--cutoffTime", help="cutoffTime", type=int, default=20)
    parser.add_argument("--suboptimality", help="suboptimality", type=float, default=2)
    parser.add_argument("--expnum", help="experiment number", type=int, default=0)
    parser.add_argument("--iter", help="iteration number", type=int)
    parser.add_argument('--firstIter', dest='firstIter', type=lambda x: bool(str2bool(x)))
    parser.add_argument('--num_parallel_runs', help="How many multiple maps in parallel tmux sessions. 1 = No parallel runs.", 
                        type=int, default=1)
    args = parser.parse_args()
    scenInputFolder = args.scenFolder
    mapsInputFolder = args.mapFolder
    firstIter = args.firstIter
    file_home = "/data_collection"
    print("iternum: " + str(args.iter))
    eecbs_runner(args)