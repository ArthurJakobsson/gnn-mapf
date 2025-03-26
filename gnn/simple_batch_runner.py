import os
import argparse
import subprocess  # For executing eecbs script
import pandas as pd  # For smart batch running
import pdb # For debugging
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For utils


mapsToMaxNumAgents = {
    "Paris_1_256": 1000, # Verified
    "random-32-32-20": 409, # Verified
    "random-32-32-10": 461, # Verified
    "den520d": 1000, # Verified
    "den312d": 1000, # Verified
    "empty-32-32": 511, # Verified
    "empty-48-48": 1000, # Verified
    "ht_chantry": 1000, # Verified
}

def getMapBDScenAgents(filepath):
    """Input: filepath to a scen or txt file
    Output: mapname, whichbd, custom_scenname, num agents
    Examples: 
    - [WHICHBD].[CUSTOMSCENNAME].[NUMAGENTS].txt 
    - [WHICHBD].[CUSTOMSCENNAME].[NUMAGENTS].scen 
    - [WHICHBD].scen, these are the default scens files 
    """
    filename = os.path.basename(filepath) # This gets the filename from the path

    splits = filename.split('.')
    assert(len(splits) == 2 or len(splits) == 4)

    whichbd = splits[0]
    mapname = whichbd.split('-')[0]

    if len(splits) == 2:
        custom_scen = splits[0]
        num_agents = 0
    else:
        custom_scen = splits[1]
        # Get num_agents
        num_agents = int(splits[2])

    return mapname, whichbd, custom_scen, num_agents

def runOnSingleInstance(pymodelArgs, numAgents, seed, scenfile):
    command = "./build_release/eecbs"
    for aKey in eecbsArgs:
        command += " --{}={}".format(aKey, eecbsArgs[aKey])
    command += " --agentNum={} --seed={} --agentsFile={}".format(numAgents, seed, scenfile)
    print(command)
    subprocess.run(command.split(" "), check=True) # True if want failure error
    
    
def getPyModelCommand(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile):
    """Command for running Python model"""
    # scenname = (scenfile.split("/")[-1])
    # mapname = mapfile.split("/")[-1].split(".")[0]
    mapname, bdname, scenname, _ = getMapBDScenAgents(scenfile)
    command = ""
    if runnerArgs["condaEnv"] is not None:
        command += "conda activate {} && ".format(runnerArgs["condaEnv"]) # e.g. conda activate pytorchfun && python -m gnn.simulator
    command += "python -m gnn.simulator"

    # Simulator parameters
    for aKey in runnerArgs["args"]:
        command += " --{}={}".format(aKey, runnerArgs["args"][aKey])
    
    command += f" --mapNpzFile=data_collection/data/benchmark_data/constant_npzs/all_maps.npz"
    command += f" --mapName={mapname} --scenFile={scenfile} --agentNum={numAgents}"
    bdFile = f"data_collection/data/benchmark_data/completed_splitting/{mapname}_bds.npz"
    command += f" --bdNpzFile={bdFile}"
    command += f" --outputCSVFile={outputfile}"
    # tempOutPath = f"{outputFolder}/paths/{scenname}{numAgents}.npy" # Note scenname ends with a .scen
    outputPathNpy = f"{outputFolder}/paths/{bdname}.{scenname}.{numAgents}.npy"
    command += f" --outputPathsFile={outputPathNpy}"
    command += f" --numScensToCreate={runnerArgs['numScensToCreate']}"
    command += f" --percentSuccessGenerationReduction={runnerArgs['percentSuccessGenerationReduction']}"
    command += f" --seed=0"
    print(command)
    subprocess.run(command.split(" "), check=True) # True if want failure error
    
def detectExistingStatus(eecbsArgs, aNum, seed, scen):
    """
    Output:
        If has been run before
        Success if run before
    """
    if not os.path.exists(eecbsArgs["output"]):
        return False, 0
    df = pd.read_csv(eecbsArgs["output"])

    ### Checks if the corresponding runs in the df have been completed already
    for aKey, aValue in eecbsArgs.items():
        ### If this is false, then we don't care about the r_weight and h_weight
        if not eecbsArgs["useWeightedFocalSearch"]:
            if aKey == "r_weight":
                continue
            if aKey == "h_weight":
                continue
        if aKey == "output":
            continue
        df = df[df[aKey] == aValue]  # Filter the dataframe to only include the runs with the same parameters
    df = df[(df["agentsFile"] == scen) & (df["agentNum"] == aNum) & (df["seed"] == seed)]
    if len(df) > 0:
        assert(len(df) == 1)
        success = (df["solution cost"] != -1).values[0]
        return True, success
    else:
        return False, 0
    
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
    
    # pymodel have different commands for inputting map, agents, and agentNum
    pymodel_map_name = mapfile.split("/")[-1].removesuffix(".map")
    assert(pymodel_map_name in mapsToMaxNumAgents.keys())
    df = df[(df["mapName"] == pymodel_map_name) & (df["scenFile"] == scenfile) & (df["agentNum"] == aNum)]
    
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

def runOnSingleMap(eecbsArgs, mapName, agentNumbers, seeds, scens):
    for aNum in agentNumbers:
        print("Starting to run {} agents on map {}".format(aNum, mapName))
        numSuccess = 0
        numToRunTotal = len(scens) * len(seeds)
        for scen in scens:
            for seed in seeds:
                runBefore, status = detectExistingStatus(eecbsArgs, aNum, seed, scen)
                if not runBefore:
                    runOnSingleInstance(eecbsArgs, aNum, seed, scen)
                    runBefore, status = detectExistingStatus(eecbsArgs, aNum, seed, scen)
                    assert(runBefore)
                numSuccess += status

        if numSuccess < numToRunTotal/4:
            print("Early terminating as only succeeded {}/{} for {} agents on map {}".format(
                                            numSuccess, numToRunTotal, aNum, mapName))
            break

def helperCreateScens(numScens, mapName, dataPath):
    scens = []
    for i in range(1, numScens+1):
        scenPath = "{}/mapf-scen-random/{}-random-{}.scen".format(dataPath, mapName, i)
        scens.append(scenPath)
    return scens


# python batch_runner.py den312d --logPath data/logs/test --cutoffTime 10 --suboptimality 2
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mapName", help="map name without .map, needs to be in mapsToMaxNumAgents defined in the top", type=str) # Note: Positional is required
    parser.add_argument("--dataPath", help="path to benchmark dataset, should contain mapf-map/ and mapf-scen-random/ folders",
                                      type=str, default="data")
    parser.add_argument("--logPath", help="path to log folder", type=str, default="data/logs/") 
    parser.add_argument("--outputCSV", help="outputCSV", type=str, default="") # Will be saved to logPath+outputCSV
    parser.add_argument("--cutoffTime", help="cutoffTime", type=int, default=60)
    parser.add_argument("--suboptimality", help="suboptimality", type=float, default=2)
    parser.add_argument("--r_weight", help="r_weight", type=float, default=4)
    parser.add_argument("--h_weight", help="h_weight", type=float, default=8)
    parser.add_argument("--num_scens", help="Number of scens to try [1,25]", type=int, default=10)
    args = parser.parse_args()

    eecbs_vs_weecsb(args)