import os
import sys
import time
from datetime import datetime # For printing current datetime
import subprocess # For executing c++ executable
import numpy as np
import argparse
import pdb
import pandas as pd
from os.path import exists
import shutil

mapsToNumAgents = {
    "Paris_1_256": (10, 260), # Verified
    "random_32_32_20": (50, 451), # Verified
    "maze_128_128_10": (10, 260),
    "den520d": (50, 1000), # Verified
    "den312d": (10, 70), # Verified
    "empty_32_32": (50, 511), # Verified
    "empty_48_48": (50, 1000), # Verified
    "ht_chantry": (10, 50), # Verified
    "random_32_32_10": (50, 451),
    "final_test0": (50, 300),
    "final_test1": (50, 300),
    "final_test2": (50, 300),
    "final_test3": (50, 300),
    "final_test4": (50, 300),
    "final_test5": (50, 300),
    "final_test6": (50, 300),
    "final_test7": (50, 300),
    "final_test8": (50, 300),
    "final_test9": (50, 300),
    "final_test10": (50, 300),
    "maze_32_32_2": (50, 91), # inc 20
    "room_32_32_4": (60, 141), # inc 20
    "warehouse_10_20_10_2_2": (50, 501) # inc 50
}

class BatchRunner:
    def __init__(self, mapfile, scenfile, timeLimit, suboptimality, batchFolderName):
        self.mapfile = mapfile
        self.scenfile = scenfile
        self.timeLimit = timeLimit
        self.suboptimality = suboptimality
        self.batchFolderName = batchFolderName

    def runSingleSettingsOnMap(self, numAgents):
        for aSeed in self.seeds:
            command = "./build_release/eecbs -m {} -a {}".format(self.mapfile, self.scenfile)
            scensplit = self.scenfile.split(".scen")[0]
            scensplit = scensplit.split("/")[-1]
            print(scensplit)
            # command += " --outputPaths={}".format("./raw_data/paths/" + "scen" + scensplit + "numAg" + str(numAgents) + "sD" + str(aSeed) + ".txt")
            # command += " --outputPaths={}".format("./raw_data/paths/" + scensplit + ".txt")
            command += " --outputPaths={}".format("./raw_data/paths/" + scensplit + str(numAgents) + str(aSeed) + ".txt") # TODO format based on instance + agents + seed
            command += " --seed={} -k {} -t {}".format(aSeed, numAgents, self.timeLimit)
            command += " --suboptimality={}".format(self.suboptimality)
            command += " --batchFolder={} -o allresults.csv".format(self.batchFolderName) 
            subprocess.run(command.split(" "), check=False) # True if want failure error
            # delete file if not max agents on the map setting
            mapname = self.mapfile.split('.')[0].split("/")[-1]
            print(mapname)
            maxagents = mapsToNumAgents[mapname][1]
            if numAgents < maxagents-1:
                # DELETE THE FILE
                os.remove("./raw_data/bd/"+ self.scenfile.split("/")[-1] + str(numAgents) + ".txt")
    
    def detectAllFailed(self, numAgents):
        """
        select all runs with the same hyperparameters (number agents, map, so)
        if a set of hyperparams fails for all seeds, stop for all future # agents
        """ 

        curOutputFile = "logs/{}/allresults.csv".format(self.batchFolderName) # TODO change allresults to more descriptive name
        if exists(curOutputFile):
            df = pd.read_csv(curOutputFile)
            df = df[(df["agent num"] == numAgents)]
            df = df[(df["suboptimality"] == self.suboptimality)]
            df = df[(df["instance name"] == self.scenfile)]
            numFailed = len(df[(df["solution cost"] <= 0) | (df["solution cost"] >= 1073741823)])
            return len(df) == numFailed
        return False
    
    def runBatchExps(self, agentNumbers, seeds):

        self.seeds = seeds
        for aNum in agentNumbers:
            print("    Number of agents: {}".format(aNum))
            self.runSingleSettingsOnMap(aNum)
            if self.detectAllFailed(aNum): # Terminate early
                print("Terminating early because all failed with {} number of agents".format(aNum))
                break

ExpSettings = dict()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="map to run", type=str) # Note: Positional is required, make not position by changing to --yKey

    args = parser.parse_args()

    mapfolder = "data/mapf-map" # TODO change to reflect where they are in this repo
    scenfolder = "data/scen-random"
    curMap = args.map # E.g. "Paris_1_256"
    if curMap not in mapsToNumAgents.keys():
        raise KeyError("Unknown map: {}".format(curMap))
    startOffset = 0
    rangeOfAgents = np.arange(mapsToNumAgents[curMap][0] + startOffset, mapsToNumAgents[curMap][1]+1, 50, dtype=int)

    myExps = []

    for aSo in [1.5]: # TODO fix; how do we treat suboptimality
        aName = "base_so{}".format(aSo)
        ExpSettings[aName] = dict(
            timeLimit=120,
            suboptimality=aSo,
        )
        myExps.append(aName)

    dateString = ""
    batchFolderName = "GridSubOptFast/GridSOFast{}_{}".format(dateString, curMap)
    
    scens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    seeds = [1, 2, 3] # TODO should we run all these scenes?
    for expName in myExps:
        print("Starting ExpSettings: {}".format(expName))
        exp = ExpSettings[expName]
        for aScenNum in scens:
            print("  Starting scen number: {}".format(aScenNum))
            
            myBR = BatchRunner("{}/{}.map".format(mapfolder, curMap), "{}/{}-random-{}.scen".format(scenfolder, curMap, aScenNum),
                exp["timeLimit"], exp["suboptimality"], batchFolderName)
            myBR.runBatchExps(rangeOfAgents, [aScenNum])

    # seeds = [1]
    # scens = [i+1 for i in range(25)] # TODO should we run all these scenes?
    # for expName in myExps:
    #     print("Starting ExpSettings: {}".format(expName))
    #     exp = ExpSettings[expName]
    #     for aScenNum in scens:
    #         print("  Starting scen number: {}".format(aScenNum))
            
    #         myBR = BatchRunner("{}/{}.map".format(mapfolder, curMap), "{}/{}-random-{}.scen".format(scenfolder, curMap, aScenNum),
    #             exp["timeLimit"], exp["suboptimality"], batchFolderName)
    #         myBR.runBatchExps(rangeOfAgents, seeds)

    # make a folder with all the current map info
    os.mkdir("./raw_data/" + curMap)
    shutil.move("./raw_data/bd/", "./raw_data/" + curMap)
    shutil.move("./raw_data/paths/", "./raw_data/" + curMap)
    
    # replace old bd and paths folder
    os.mkdir("./raw_data/bd")
    os.mkdir("./raw_data/paths") 

if __name__ == "__main__":
    main()