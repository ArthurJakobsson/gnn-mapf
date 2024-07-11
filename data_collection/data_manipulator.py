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
import torchvision
from torch.utils.data import Dataset
from collections import defaultdict
import pdb
import math
import random

'''
0. parse bd (fix eecbs by having it make another txt output, parse it here)
    note that each agent has its own bd for a given map
1. make it a batch that accepts directories instead of files
2. dictionary that maps map_name to map
3. dictionary that maps bd_name to bd
4. write tuples of (map name, bd name, paths np) to npz
'''


class PipelineDataset(Dataset):
    '''
    A dataset loader that allows you to store eecbs instances.
    '''

    # instantiate class variables
    def __init__(self, numpy_data_path, k, size, max_agents, helper_bd_preprocess="middle"):
        '''
        INPUT:
            numpy_data_path: the path to the npz file storing all map, backward dijkstra, and path information across all EECBS runs. (string)
            k: window size. (int)
        contains 3 class variables: self.maps, self.bds, and self.tn2.
        maps: a dictionary mapping mapname (name.map) to the np (w,h) of all obstacle locations. (1s for obstacles, 0s for free space)
            e.g. {"Paris_1_256.map": (256,256)}
        bds: a dictionary mapping backward dijkstra names to the np (n,w,h) of all backward dijkstra calculations for all agents.
            e.g. {"Paris_1_256-random-1{max_agents}": ({max_agents},256,256)}
            naming convention: scen name + # agents
        tn2: a dictionary mapping eecbs instance names to the np (t,n,2) of all paths (x,y) for all agents.
            e.g. {"Paris_1_256.map,Paris_1_256-random-110,2": (t,n,2)}
            naming convention: mapname + "," + bdname + "," + seed
        helper_bd_preprocess: method by which we center helper backward dijkstras. can be 'middle', 'current', or 'subtraction'.
        '''

        # read in the dataset, saving map, bd, and path info to class variables
        loaded = np.load(numpy_data_path)
        self.k = k
        self.size = size
        self.parse_npz(loaded) # TODO change len(dataloader) = max_timesteps
        self.max_agents = max_agents
        self.helper_bd_preprocess = helper_bd_preprocess


    # get number of instances in total (length of training data)
    def __len__(self):
        return self.length # go through the tn2 dict with # data and np arrays saved, and sum all the ints

    # return the data for a particular instance: the location, bd, and map
    def __getitem__(self, idx):
        '''
        INPUT: index (must be smaller than len(self))
        OUTPUT: map, bd, and direction
            map: (2k+1, 2k+1)
            bd: (2k+1, 2k+1)
            other agent bds: (4,2k+1,2k+1)
            direction: (2)
        centered version. when passing in the map and bd, return a (2k+1,2k+1) window centered at current location of agent.
        '''
        if idx >= self.__len__():
            print("Index too large for {}-sample dataset".format(self.__len__()))
            return
        bd, grid, paths, timestep, t = self.find_instance(idx)
        labels = []
        locs = []
        num_agent = paths.shape[1]
        for agent in range(0,num_agent): # TODO: numpyify this
            curloc = paths[timestep, agent]
            nextloc = paths[timestep+1, agent] if timestep < t-1 else curloc
            label = nextloc - curloc # get the label: where did the agent go next?
            # create one-hot vector
            index = None
            if label[0] == 0 and label[1] == 0: index = 0
            elif label[0] == 0 and label[1] == 1: index = 1
            elif label[0] == 1 and label[1] == 0: index = 2
            elif label[0] == -1 and label[1] == 0: index = 3
            else: index = 4
            finallabel = np.zeros(5)
            finallabel[index] = 1
            labels.append(finallabel)
            locs.append(curloc)
        return np.array(locs), np.array(labels), bd, grid

    def process_helper_bds(self, bd, windowAgentLocs, windowAgents, curloc, dijk):
        '''
        input: bd
        from class construction we'll pass in a flag representing how to preprocess the helper_bds (self.helper_bd_preprocess)
        if this flag is 'middle', normalize by the value contained at center of the bd
        if this flag is 'current', normalize by where the helper bd's agent is
        if this flag is 'subtraction', normalize by subtracting current agent's bd from 'current'
        note that all bds will still be centered at the current agent's location.
        '''

        # for each agent in window, get its bd, centered at curloc with provided window size
        helper_bds = [bd[inwindow[1]][curloc[0]-self.k:curloc[0]+self.k+1, curloc[1]-self.k:curloc[1]+self.k+1] for inwindow in windowAgents] # for each of the (at most) 4 nearby agents, get their bds centered at the current agent's location
        helper_bds = np.array(helper_bds)
        # for each helper bd, subtract out the middle TODO validate
        helper_bds = np.array([helper_bd - bd[inwindow[1]][loc[0]][loc[1]] for inwindow, loc, helper_bd in zip(windowAgents, windowAgentLocs, helper_bds)])

        # helper_bds = np.where(helper_bds == 1073741823, 0, helper_bds)

        # normalization logic
        # if len(helper_bds) >= 1: pdb.set_trace()
        # if self.helper_bd_preprocess == 'current':
        #     # if len(helper_bds) >= 1: pdb.set_trace()
        #     helper_bds = np.array([helper_bd - bd[inwindow[1]][loc[0]][loc[1]] for inwindow, loc, helper_bd in zip(windowAgents, windowAgentLocs, helper_bds)])
        #     helper_bds = np.where(helper_bds > 1000000000, 0, helper_bds) # TODO np.where after subtraction (probably, turn anything smaller than -a million to filler val)
        # elif self.helper_bd_preprocess == 'subtraction':
        #     helper_bds = np.array([bd - dijk for bd in helper_bds])
        #     # assert(np.all(abs(helper_bds) < 1073741823))
        # else: # default to 'middle'
        #     helper_bds = np.array([bd - bd[self.k,self.k] for bd in helper_bds])
        #     helper_bds = np.where(helper_bds > 1000000000, 0, helper_bds) # TODO np.where after subtraction (probably, turn anything smaller than -a million to filler val)

        # pad empty entries with 0s
        n = len(helper_bds)
        if n < 4:
            if n == 0:
                windowAgentLocs = np.array([[curloc[0], curloc[1]]]*4)
                helper_bds = np.zeros((4-n, self.k*2+1,self.k*2+1))
            else:
                windowAgentLocs = np.concatenate([windowAgentLocs, np.array([[curloc[0], curloc[1]]]*(4-n))])
                helper_bds = np.concatenate([helper_bds, np.zeros((4-n, self.k*2+1,self.k*2+1))])

        return helper_bds, windowAgentLocs


    def find_instance(self, idx):
        '''
        returns the backward dijkstra, map, and path arrays, and indices to get into the path array
        '''
        def translate_bd_name(bdname):
            if "-custom-" in bdname:
                split_name = bdname.split("-custom-")
                front = split_name[0] # scen name
                back = (split_name[1].split("-"))[-1] # number of agents
                bdname = front+back
            return bdname

        items = list(self.tn2.items())

        tn2ind = 0
        tracker = 0
        while tracker + items[tn2ind][1][0] <= idx:
            tracker += items[tn2ind][1][0] # add number of data in the (t,n,2) matrix
            tn2ind += 1
        # so now tn2ind holds the index to the (t,n,2) matrix containing the data we want
        mapname, bdname = items[tn2ind][0].split(",")
        bdname = translate_bd_name(bdname)
        bd = self.bds[bdname]      # (N, W, H) # TODO: need to chop of the "custom" part
        grid = self.maps[mapname]  # (W, H)
        # pad bds (for all agents), grid (for all agents) with empty 0 window(s), k in all directions

        # get the location, dir to next location
        newidx = idx - tracker # index within the matrix to get
        paths = items[tn2ind][1][1].copy() # (t,n,2) paths matrix
        paths += self.k # adjust for padding
        bd *= (1-grid) 
        t, n, _ = np.shape(paths)
        timestep = newidx // n
        return bd, grid, paths, timestep, t

    def parse_npz(self, loaded):
        loaded = {k:v for k, v in loaded.items()}
        items = list(loaded.items())
        # print(loaded["Paris_1_256.map,Paris_1_256-random-110,2"]) # testing
        # index -> tuple mapping, finding maps, then bds, then paths
        i = 0
        while "-random-" not in items[i][0]:
            i += 1
        self.maps = dict(items[:i]) # get all the maps
        j = i
        while "," not in items[j][0]:
            j += 1
        self.bds = dict(items[i:j]) # get all the bds
        k = j
        while k < len(items) and "twh" not in items[k][0]:
            k += 1
        self.tn2 = dict(items[j:k]) # get all the paths in (t,n,2) form
        # since the # of data is simply number of agent locations, this is t*n, which we append to the dictionary for each path
        totalT = 0 
        for ky, v in self.tn2.items():
            t, n, _ = np.shape(v)
            self.tn2[ky] = (t*n, v)
            totalT += t
        self.length = totalT # number of paths = number of timesteps
        # self.twh = dict(items[k:]) # get all the paths in (t,w,h) form
        npads = ((0,0),(self.k, self.k), (self.k, self.k))
        for key in self.bds:
            self.bds[key] = np.pad(self.bds[key], npads, mode="constant", constant_values=1073741823)
            # self.bds[key] = np.transpose(self.bds[key], (0, 2, 1)) # (n,h,w) -> (n,w,h) NOTE that originally all bds are parsed in transpose TODO did i fix this correctly
        for key in self.maps:
            self.maps[key] = np.pad(self.maps[key], self.k, mode="constant", constant_values=1)


def parse_map(mapfile):
    '''
    takes in a mapfile and returns a parsed np array
    '''
    with open(mapfile) as f:
        line = f.readline()  # "type octile"

        line = f.readline()  # "height 32"
        height = int(line.split(' ')[1])

        line = f.readline()  # width 32
        width = int(line.split(' ')[1])

        line = f.readline()  # "map\n"
        assert(line == "map\n")

        mapdata = np.array([list(line.rstrip()) for line in f])

    mapdata.reshape((width,height))
    mapdata[mapdata == '.'] = 0
    mapdata[mapdata == '@'] = 1
    mapdata[mapdata == 'T'] = 1
    mapdata = mapdata.astype(int)
    return mapdata

def parse_path(pathfile):
    '''
    reads a txt file of paths for each agent, returning a dictionary mapping timestep->position of each agent
    inputs: pathfile (string)
    outputs: (T,N,2) np.darray: where is each agent at time T?
    '''
    # save dimensions for later array saving
    w = h = 0
    # maps timesteps to a list of agent coordinates
    timestepsToMaps = defaultdict(list)
    # get max number of timesteps by counting number of commas
    maxTimesteps = 0
    with open(pathfile, 'r') as fd:
        linenum = 0
        for line in fd.readlines():
            if linenum == 0:
                linenum += 1
                continue # ignore dimension line
            timesteps = 0
            for c in line:
                if c == ',': timesteps += 1
            maxTimesteps = max(maxTimesteps, timesteps)
            linenum += 1

    # get path for each agent and update dictionary of maps accordingly
    with open(pathfile, 'r') as fd:
        linenum = 0
        for line in fd.readlines():
            if linenum == 0: # parse dimensions of map and keep going
                line = line[:-1]
                line = line.split(",")
                w = int(line[0])
                h = int(line[1])
                linenum += 1
                continue
            i = 0
            # omit up to the first left paren
            while line[i] != '(': i += 1
            # omit the ending space and final arrow to nothing
            line = line[i:-3]
            # get list of coordinates, as raw (x, y) strings
            rawCoords = line.split("->")
            # add the coordinates to the dictionary of maps
            for i, coord in enumerate(rawCoords):
                temp = coord.split(',')
                x = int(temp[0][1:])
                y = int(temp[1][:-1])
                # if you're at the last coordinate then append it to the rest of the maps
                if i == len(rawCoords) - 1:
                    while i != maxTimesteps:
                        timestepsToMaps[i].append([x, y])
                        i += 1
                else: timestepsToMaps[i].append([x, y])
            linenum += 1

    # make each map a np array
    for key in timestepsToMaps:
        timestepsToMaps[key] = np.asarray(timestepsToMaps[key])

    # make this t x n x 2
    res = []
    for i in range(len(timestepsToMaps)):
        res.append(timestepsToMaps[i])

    t, n = len(res), len(res[0])

    # TODO and then make a t x w x h and return that too
    res2 = np.zeros((t, w, h))
    res2 -= 1 # if no agent, -1
    for time in range(t):
        arr = res[time]
        for agent in range(n):
            width, height = arr[agent]
            res2[time][width][height] = agent

    res = np.asarray(res)
    # print(t, w, h, n)

    # res2 = [[[1 if [width, height] in res[time] else 0 for width in range(w)] for height in range(h)] for time in range(t)]
    return res

def parse_bd(bdfile):
    '''
    parses a txt file of bd info for each agent
    input: bdfile (string)
    output: (N,H,W) NOTE: this is a transposed bd compared to the map! (fixed in npz parsing logic in dataloader)
    '''
    timetobd = defaultdict(list)
    w, h = None, None
    with open(bdfile, 'r') as fd:
        agent = 0
        linenum = 0
        for line in fd.readlines():
            if linenum == 0: # parse dimensions and keep going
                line = line[:-1]
                line = line.split(",")
                h = int(line[0])
                w = int(line[1])
                linenum += 1
                continue
            line = line[:-2]
            heuristics = line.split(",")
            heuristics = [int(x) for x in heuristics]
            timetobd[agent] = heuristics
            agent += 1
    for key in timetobd:
        timetobd[key] = np.asarray(timetobd[key])
        nwh = timetobd[key]
        new = []
        assert(not len(nwh) % w and not len(nwh) % h)
        # transform to n x w x h here, assuming row-major order
        while len(nwh):
            takeaway = nwh[:w]
            new.append(takeaway)
            nwh = nwh[w:]
        timetobd[key] = new

    # make this n x w x h from dictionary of n w x h arrays
    res = []
    for i in range(len(timetobd)):
        res.append(timetobd[i])
    res = np.asarray(res)
    return res

def batch_map(dir):
    '''
    goes through a directory of maps, parsing each one and saving to a dictionary
    input: directory of maps (string)
    output: dictionary mapping filenames to parsed maps
    '''

    res = {} # string->np
    # iterate over files in directory, parsing each map
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        print(f)
        # checking if it is a file
        if os.path.isfile(f):
            if ".DS_Store" in f: continue # deal with invisible ds_store file
            # parse the map file and add to a global dictionary (or some class variable dictionary)
            val = parse_map(f)
            res[filename] = val
        else:
            raise RuntimeError("bad map dir")
    return res

def batch_bd(dir):
    '''
    goes through a directory of bd outputs, parsing each one and saving to a dictionary
    input: directory of backward djikstras (string)
    output: dictionary mapping filenames to backward djikstras
    '''
    res = {} # string->np
    # iterate over files in directory, parsing each map
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # parse the bd file and add to a global dictionary (or some class variable dictionary)
            val = parse_bd(f)
            bdname, agents = (filename.split(".txt")[0]).split(".scen") # e.g. "Paris_1_256-random-110, where 1 is instance, 10 is agents"
            res[bdname + agents] = val # TODO make sure that filename doesn't have weird chars you don't want in the npz
            print(f)
        else:
            raise RuntimeError("bad bd dir")
    return res

def batch_path(dir):
    '''
        goes through a directory of outputted EECBS paths,
        returning a dictionary of tuples of the map name, bd name, and paths dictionary
        NOTE we assume that the file of each path is formatted as 'raw_data/paths/mapnameandbdname.txt'
        NOTE and also that bdname has agent number grandfathered into it
    '''
    res1 = {} # dict of (mapname, bdname, int->np.darray dictionary), and is (n, t, 2); train set only
    # res2 = {} # dict of (mapname, bdname, int->np.darray dictionary), and is (t, w, h)
    res2 = {} # dict of the same thing, but for val set
    numFiles = 0
    # get number of files
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if os.path.isfile(f):
            numFiles += 1
        else:
            raise RuntimeError("bad path dir")
    # valFiles = random.sample(range(0, numFiles), numFiles // 5) # take 20% of data for val
    # print(valFiles, numFiles)
    valFiles = [] # No more validation dataset

    # iterate over files in directory, making a tuple for each
    idx = 0 # keep track of if we want to save to val set or train set
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # parse the path file, index out its map name, seed, agent number, and bd that it was formed from based on file name,
            # and add the resulting triplet to a global dictionary (or some class variable dictionary)
            raw = filename.split(".txt")[0].split(".scen") # remove .txt
            raw = raw[0]+raw[1]
            # raw = raw[:-1]
            mapname = raw.split("-")[0] + ".map"
            bdname = raw
            val = parse_path(f) # get the 2 types of paths: the first being a list of agent locations for each timestep, the 2nd being a map for each timestep with -1 if no agent, agent number otherwise
            # print(mapname, bdname, seed, np.count_nonzero(val2 != -1)) # debug statement
            print("___________________________\n")
            if idx in valFiles:
                res2[mapname + "," + bdname] = val
            else:
                res1[mapname + "," + bdname] = val
            # res2[mapname + "," + bdname + "," + seed + ",twh"] = val2
            print(f)
            idx += 1
        else:
            raise RuntimeError("bad path dir")
    return res1, res2

def main():
    # cmdline argument parsing: take in dirs for paths, maps, and bds, and where you want the outputted npz
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathsIn", help="directory containing txt files of agents and paths taken", type=str)
    parser.add_argument("--bdIn", help="directory containing txt files with backward djikstra output", type=str)
    parser.add_argument("--mapIn", help="directory containing txt files with obstacles", type=str)
    npzMsg = "output file with maps, bds as name->array dicts, along with (mapname, bdname, path) triplets for each EECBS run"
    parser.add_argument("--trainOut", help=npzMsg, type=str)
    parser.add_argument("--valOut", help=npzMsg, type=str)
    

    args = parser.parse_args()

    pathsIn = args.pathsIn
    bdIn = args.bdIn
    mapIn = args.mapIn
    trainOut = args.trainOut
    valOut = args.valOut

    # instantiate global variables that will keep track of each map and bd that you've encountered
    maps = {} # maps mapname->np array containing the obstacles in map
    bds = {} # maps bdname->np array containing bd for each agent in the instance (NOTE: keep track of number agents in bdname)

    # parse each map, add to global dict
    
    maps = batch_map(mapIn)
    # print(maps)

    # parse each bd, add to global dict
    bds = batch_bd(bdIn)
    # print(bds)

    # parse each path, add to global list
    data1train, data1val = batch_path(pathsIn)

    # send each map, each bd, and each tuple representing a path + instance to npz
    np.savez_compressed(trainOut, **maps, **bds, **data1train) # Note automatically stacks to numpy vectors

    # DEBUGGING: test out the dataloader
    loader = PipelineDataset(trainOut + ".npz", 4, float('inf'), 300, 'current')
    print(len(loader), " train size")
    for i in range(len(loader)):
        locs, labels, bd, grid = loader[i]
        assert(labels.shape[1] == 5)
        assert(locs.shape[1] == 2)
        # grid/bd should be 9,9 for single agent but is full map for graph version



if __name__ == "__main__":
    main()
