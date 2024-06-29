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


mapsToMaxNumAgents = {
    "Paris_1_256": 1000, # Verified
    "random-32_32_20": 409, # Verified
    "random_32_32_10": 461, # Verified
    "den520d": 1000, # Verified
    "den312d": 1000, # Verified
    "empty-32-32": 511, # Verified
    "empty-48-48": 1000, # Verified
    "ht_chantry": 1000, # Verified
    "warehouse_10_20_10_2_2": 101,
    "warehouse_20_40_10_2_2": 101
}

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def runOnSingleInstance(eecbsArgs, numAgents, seed, scenfile):
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

    tempOutPath = f".{file_home}/eecbs/raw_data/paths/{scenfile}{numAgents}.txt"
    command += " --agentNum={} --seed={} --agents={} --outputPaths={} --firstIter={}".format(numAgents, seed, scenfile, tempOutPath, firstIter)
    print(command)
    subprocess.run(command.split(" "), check=True) # True if want failure error
    
    
def detectExistingStatus(eecbsArgs, aNum, seed, scen): # TODO update
    """
    Output:
        If has been run before
        Success if run before
    """
    if not os.path.exists(eecbsArgs["outputPaths"]):
        print("here2")
        return False, 0
    print('here3')
    # TODO: fix broken logic... revolves around csv never being created
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

def runOnSingleMap(eecbsArgs, mapName, agentNumbers, seeds, scens, inputFolder):
    print("Single Map")
    if "benchmark" in inputFolder:
        for aNum in agentNumbers:
            print("Starting to run {} agents on map {}".format(aNum, mapName))
            numSuccess = 0
            status=0
            numToRunTotal = len(scens) * len(seeds)
            for scen in scens:
                for seed in seeds:
                    # runBefore, status = detectExistingStatus(eecbsArgs, aNum, seed, scen)
                    runBefore=False #TODO fix detectExisting
                    if not runBefore:
                        print(scen)
                        runOnSingleInstance(eecbsArgs, aNum, seed, scen)
                        # runBefore, status = detectExistingStatus(eecbsArgs, aNum, seed, scen)
                        # assert(runBefore)
                        status+=1
                    numSuccess += status

            if numSuccess < numToRunTotal/2:
                print("Early terminating as only succeeded {}/{} for {} agents on map {}".format(
                                                numSuccess, numToRunTotal, aNum, mapName))
                break
    else:
        for aNum, scen in zip(agentNumbers, scens): # for each scen, run its agent number
            print("Starting to run {} agents on map {}".format(aNum, mapName))
            for seed in seeds:
                # runBefore, status = detectExistingStatus(eecbsArgs, aNum, seed, scen)
                runBefore=False #TODO fix detectExisting
                if not runBefore:
                    runOnSingleInstance(eecbsArgs, aNum, seed, scen)
                    # runBefore, status = detectExistingStatus(eecbsArgs, aNum, seed, scen)
                    # assert(runBefore)

# def helperCreateScens(numScens, mapName, dataPath):
#     scens = []
#     for i in range(1, numScens+1):
#         scenPath = "{}/mapf-scen-random/{}-random-{}.scen".format(dataPath, mapName, i)
#         scens.append(scenPath)
#     return scens

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
        mapsToScens[mapFile].append(dir_path)

    for mapFile in mapsToScens:
        eecbsArgs = {
            "map": f"{mapsInputFolder}/{mapFile}.map",
            "output": f".{file_home}/eecbs/raw_data/bd/stats.csv",
            # "outputPaths": f".{file_home}/eecbs/raw_data/paths/paths.txt",
            "suboptimality": args.suboptimality,
            "cutoffTime": args.cutoffTime
            # "useWeightedFocalSearch": False,
        }
        seeds = list(range(1,2))
        # scens = helperCreateScens(args.num_scens, args.mapName, args.dataPath)
        scens = mapsToScens[mapFile]

        if "benchmark" in scenInputFolder: # pre-loop run
            increment = 100
            agentNumbers = list(range(increment, mapsToMaxNumAgents[mapFile]+1, increment))

            ### Run baseline EECBS
            runOnSingleMap(eecbsArgs, mapFile, agentNumbers, seeds, scens, scenInputFolder)

        else: # we are somewhere in the training loop
            agentNumbers = []
            scenlist = os.listdir(scenInputFolder)
            for scen in scenlist:
                # open the file
                with open(f'{scenInputFolder}/{scen}', 'r') as fh: # TODO get the path to scene file right
                    line = fh.readline()
                    agentNumbers.append(int(line))
            # run eecbs
            runOnSingleMap(eecbsArgs, mapFile, agentNumbers, seeds, scens, scenInputFolder)

        # move the new eecbs output
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}", exist_ok=True)
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/bd/", exist_ok=True)
        os.makedirs(f".{file_home}/eecbs/raw_data/{mapFile}/paths/", exist_ok=True)

        bd_files = os.listdir(f".{file_home}/eecbs/raw_data/bd/")
        path_files = os.listdir(f".{file_home}/eecbs/raw_data/paths/")
        for file_name in bd_files:
            shutil.move(os.path.join(f".{file_home}/eecbs/raw_data/bd/", file_name), f".{file_home}/eecbs/raw_data/{mapFile}/bd/{file_name}") 
        for file_name in path_files:
            shutil.move(os.path.join(f".{file_home}/eecbs/raw_data/paths/", file_name), f".{file_home}/eecbs/raw_data/{mapFile}/paths/{file_name}") 
        
        # shutil.move(f".{file_home}/eecbs/raw_data/bd/", f".{file_home}/eecbs/raw_data/" + mapFile)
        # shutil.move(f".{file_home}/eecbs/raw_data/paths/", f".{file_home}/eecbs/raw_data/{mapFile}")
        # os.mkdir(f".{file_home}/eecbs/raw_data/paths/")
        

        # make the npz files
        # TODO don't hardcode exp, iter numbers
        pdb.set_trace()
        subprocess.run(["python", "./data_collection/data_manipulator.py", f"--pathsIn=.{file_home}/eecbs/raw_data/{mapFile}/paths/", f"--bdIn=.{file_home}/eecbs/raw_data/{mapFile}/bd/", f"--mapIn={mapsInputFolder}", f"--trainOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/train_{mapFile}_{args.iter}", f"--valOut=.{file_home}/data/logs/EXP{args.expnum}/labels/raw/val_{mapFile}_{args.iter}"])
        

        # os.mkdir(f".{file_home}/eecbs/raw_data/bd")
        # os.mkdir(f".{file_home}/eecbs/raw_data/paths")
        shutil.rmtree(f".{file_home}/eecbs/raw_data/{mapFile}/paths") # clean path files once they have been recorded
        os.mkdir(f".{file_home}/eecbs/raw_data/{mapFile}/paths") # remake "path" folder
        ### Run W-EECBS
        # eecbsArgs["r_weight"] = args.r_weight
        # eecbsArgs["h_weight"] = args.h_weight
        # eecbsArgs["useWeightedFocalSearch"] = True
        # runOnSingleMap(eecbsArgs, args.mapName, agentNumbers, seeds, scens)

    ### Load in the data
    # df = pd.read_csv(totalOutputPath)
    # # Select only those with the correct cutoff time and suboptimality
    # df = df[(df["cutoffTime"] == args.cutoffTime) & (df["suboptimality"] == args.suboptimality)]

    # dfRegEECBS = df[df["useWeightedFocalSearch"] == False]
    # dfWEECBS = df[(df["useWeightedFocalSearch"] == True) & (df["r_weight"] == args.r_weight) & (df["h_weight"] == args.h_weight)]

    # ### Compare the relative speed up when the num agents and seeds are the same
    # df = pd.merge(dfRegEECBS, dfWEECBS, on=["agentNum", "seed", "agentsFile"], how="inner", suffixes=("_reg", "_w"))
    # df = df[(df["solution cost_reg"] != -1) & (df["solution cost_w"] != -1)] # Only include the runs that were successful
    # df["speedup"] = df["runtime_reg"] / df["runtime_w"]

    # ### Plot speed up of W-EECBS over EECBS for each agent number
    # df.boxplot(column="speedup", by="agentNum", grid=False) # create botplot for each agent number
    # plt.axhline(y=1, color='k', linestyle='--', alpha=0.5) # Add a line at y=1
    # plt.suptitle('')
    # plt.title("W-EECBS w_so={} r={} w_h={} on {}".format(args.suboptimality, args.r_weight, args.h_weight, args.mapName))
    # plt.xlabel("Number of agents")
    # plt.ylabel("EECBS/W-EECBS Runtime Ratio")
    # plt.savefig("{}/{}_weecbsSo{}R{}H{}_speedup.pdf".format(args.logPath, args.mapName, args.suboptimality, args.r_weight, args.h_weight))


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
    args = parser.parse_args()
    scenInputFolder = args.scenFolder
    mapsInputFolder = args.mapFolder
    firstIter = args.firstIter
    file_home = "/data_collection"
    print("iternum: " + str(args.iter))
    eecbs_runner(args)