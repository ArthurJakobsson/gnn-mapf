import os
import sys
import time
from datetime import datetime # For printing current datetime
import subprocess # For executing c++ executable
import numpy as np
import argparse
import pdb
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict
import multiprocessing
# print(os.path.abspath(os.getcwd()))
# sys.path.insert(0, './data_collection/')
# sys.path.append(os.path.abspath(os.getcwd())+"/custom_utils/")
from custom_utils.custom_timer import CustomTimer

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
    def __init__(self, mapFileNpz, bdFileNpz, pathFileNpz, k, size, max_agents, helper_bd_preprocess="middle"):
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
        # raw_folder = numpy_data_path.split("train_")[0]
        # loaded_maps = np.load(raw_folder+"maps.npz")
        # loaded_bds = np.load(numpy_data_path.split(".npz")[0]+"_bds.npz")
        # read in the dataset, saving map, bd, and path info to class variables
        # loaded = np.load(numpy_data_path)
        # self.numpy_data_path = numpy_data_path
        assert(mapFileNpz.endswith(".npz") and bdFileNpz.endswith(".npz") and pathFileNpz.endswith(".npz"))
        self.maps = dict(np.load(mapFileNpz))
        self.bds = dict(np.load(bdFileNpz))
        self.tn2 = dict(np.load(pathFileNpz)) # Note: Very important to make this a dict() otherwise lazy loading kills performance later on
        self.k = k
        self.size = size
        # self.parse_npz(loaded) # TODO change len(dataloader) = max_timesteps
        self.max_agents = max_agents
        self.helper_bd_preprocess = helper_bd_preprocess

        self.parse_npz2()


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
        bd, grid, paths, timestep, max_timesteps = self.find_instance(idx)
        # labels = []
        # locs = []
        # num_agent = paths.shape[1]
        # for agent in range(0,num_agent): # TODO: numpyify this
        #     curloc = paths[timestep, agent]
        #     nextloc = paths[timestep+1, agent] if timestep < t-1 else curloc
        #     label = nextloc - curloc # get the label: where did the agent go next?
        #     # create one-hot vector
        #     index = None
        #     if label[0] == 0 and label[1] == 0: index = 0
        #     elif label[0] == 0 and label[1] == 1: index = 1
        #     elif label[0] == 1 and label[1] == 0: index = 2
        #     elif label[0] == -1 and label[1] == 0: index = 3
        #     else: index = 4
        #     finallabel = np.zeros(5)
        #     finallabel[index] = 1
        #     labels.append(finallabel)
        #     locs.append(curloc)
        # return np.array(locs), np.array(labels), bd, grid
        cur_locs = paths[timestep] # (N,2)
        next_locs = paths[timestep+1] if timestep+1 < max_timesteps else cur_locs # (N,2)
        deltas = next_locs - cur_locs # (N,2)

        # Define the mapping from direction vectors to indices
        direction_labels = np.array([(0,0), (0,1), (1,0), (-1,0), (0,-1)]) # (5,2)
        # Find the index of each direction in the possible_directions array
        indices = np.argmax(np.all(deltas[:, None] == direction_labels, axis=2), axis=1)
        # Create a one-hot encoded array using np.eye
        labels = np.eye(direction_labels.shape[0])[indices]
        return cur_locs, labels, bd, grid


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

        assert(idx < self.length)
        # pdb.set_trace()
        # items = list(self.tn2.items())
        total_sum = 0
        key_to_use = None
        timestep_to_use = 0
        for aKey, pathVec in self.tn2.items(): #pathVec is (T,N,2)
            timesteps = pathVec.shape[0]
            if total_sum + timesteps > idx:
                key_to_use = aKey
                timestep_to_use = idx - total_sum
                break
            total_sum += timesteps
        assert(key_to_use is not None)

        if len(key_to_use.split(",")) != 3:
            raise KeyError(f"Badly formatted paths key: {key_to_use}, should check parse_paths and \
                            make sure that keys are in mapName,scenName,numAgents format")
        mapname, scenname, num_agents = key_to_use.split(",")
        num_agents = int(num_agents)
        # bdname = translate_bd_name(bdname)
        # scenname, num_agents = bdname.split(",")
        # bd = self.bds[bdname]      # (N,W,H) # TODO: need to chop of the "custom" part
        grid = self.maps[mapname]  # (W,H)
        paths = self.tn2[key_to_use] + self.k # (T,N,2)
        max_timesteps = paths.shape[0]
        assert(self.bds[scenname].shape[0] >= num_agents)
        bd = self.bds[scenname][:num_agents] # (N,W,H)
        # pad bds (for all agents), grid (for all agents) with empty 0 window(s), k in all directions

        # get the location, dir to next location
        # newidx = idx - tracker # index within the matrix to get
        # paths = items[tn2ind][1][1].copy() # (t,n,2) paths matrix
        # paths += self.k # adjust for padding
        # bd *= (1-grid) 
        # t, n, _ = np.shape(paths)
        # timestep = newidx // n
        # return bd, grid, paths, timestep, t

        return bd, grid, paths, timestep_to_use, max_timesteps

    def parse_npz(self, loaded_paths, loaded_maps, loaded_bds):
        self.tn2 = {k:v for k, v in loaded_paths.items()}
        self.maps = {k:v for k, v in loaded_maps.items()}
        self.bds = {k:v for k, v in loaded_bds.items()}

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

    def parse_npz2(self):
        totalT = 0 
        for ky, v in self.tn2.items():
            t, n, _ = np.shape(v)
            totalT += t
        self.length = totalT

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
    outputs: (T,N,2) np.darray: where each agent is at time T
    TODO: NUMPYIFY THIS
    '''
    # save dimensions for later array saving
    # pdb.set_trace()
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

    # t, n = len(res), len(res[0])

    # TODO and then make a t x w x h and return that too
    # res2 = np.zeros((t, w, h))
    # res2 -= 1 # if no agent, -1
    # for time in range(t):
    #     arr = res[time]
    #     for agent in range(n):
    #         width, height = arr[agent]
    #         res2[time][width][height] = agent

    res = np.asarray(res) # (t,n,2)
    return res

def parse_bd(bdfile):
    '''
    parses a txt file of bd info for each agent
    input: bdfile (string)
    output: (N,H,W) NOTE: this is a transposed bd compared to the map! (fixed in npz parsing logic in dataloader)
    '''
    agent_to_bd = []
    w, h = None, None
    with open(bdfile, 'r') as fd:
        for i, line in enumerate(fd.readlines()):
            if i == 0: # parse dimensions and keep going
                line = line[:-1]
                line = line.split(",")
                h = int(line[0])
                w = int(line[1])
            else:
                agent_to_bd.append(np.fromstring(line, dtype=int, sep=',')) # (W*H+1)
    agent_to_bd = np.asarray(agent_to_bd)[:,:-1] # (N, W*H) removes trailing comma
    agent_to_bd = agent_to_bd.reshape((agent_to_bd.shape[0], h, w)) # (N, H, W)

    # tmp = np.fromstring(allLines, dtype=int, sep=',') # Ideally, but fails due to trailing comma
    # tmp = np.genfromtxt(bdfile, delimiter=',', dtype=int, skip_header=1) # Works but too slow

    return agent_to_bd

def batch_map(dir, num_parallel):
    '''
    goes through a directory of maps, parsing each one and saving to a dictionary
    input: directory of maps (string)
    output: dictionary mapping filenames to parsed maps
    '''

    res = {} # string->np
    inputs_list = []
    filenames_list = []
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if os.path.isfile(f):
            if ".DS_Store" in f: continue # deal with invisible ds_store file
            # parse the map file and add to a global dictionary (or some class variable dictionary)
            if num_parallel == 1: # Parse it directly
                res[filename] = parse_map(f)
            else:
                inputs_list.append((f,))  # Note, need to pass in as tuple for use with starmap
                filenames_list.append(filename)
        else:
            raise RuntimeError("bad map dir")
        
    if num_parallel == 1:
        return res
    
    with multiprocessing.Pool(processes=num_parallel) as pool:
        results = pool.starmap(parse_map, inputs_list)

    for i in range(len(inputs_list)):
        filename = filenames_list[i]
        res[filename] = results[i]
        
    return res

def batch_bd(dir, num_parallel):
    '''
    goes through a directory of bd outputs, parsing each one and saving to a dictionary
    input: directory of backward djikstras (string)
    output: dictionary mapping filenames to backward djikstras
    '''
    res = {} # string->np
    inputs_list = []
    filenames_list = []
    # iterate over files in directory, parsing each map
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # parse the bd file and add to a global dictionary (or some class variable dictionary)
            # val = parse_bd(f)
            scenname, agents = (filename.split(".txt")[0]).split(".scen") # e.g. "Paris_1_256-random-110, where 1 is instance, 10 is agents"
            if num_parallel == 1:
                res[scenname] = parse_bd(f)
            else:
                inputs_list.append((f,))  # Note, need to pass in as tuple for use with starmap
                filenames_list.append(scenname)
        else:
            raise RuntimeError("bad bd dir")
    
    if num_parallel == 1:
        return res

    with multiprocessing.Pool(processes=num_parallel) as pool:
        results = pool.starmap(parse_bd, inputs_list)

    for i in range(len(inputs_list)):
        filename = filenames_list[i]
        res[filename] = results[i]
    return res

def batch_path(dir):
    '''
        goes through a directory of outputted EECBS paths,
        returning a dictionary of tuples of the map name, bd name, and paths dictionary
        NOTE we assume that the file of each path is formatted as 'raw_data/paths/mapnameandbdname.txt'
        NOTE and also that bdname has agent number grandfathered into it
    '''
    res1 = {} # dict of (mapname, bdname, int->np.darray dictionary), and is (n, t, 2);
    numFiles = 0
    # get number of files
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if os.path.isfile(f):
            numFiles += 1
        else:
            raise RuntimeError("bad path dir")
    # iterate over files in directory, making a tuple for each
    idx = 0 # keep track of if we want to save to val set or train set
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # parse the path file, index out its map name, seed, agent number, and bd that it was formed from based on file name,
            # and add the resulting triplet to a global dictionary (or some class variable dictionary)
            # Example filename: 'empty_8_8-random-9.scen32.txt'
            scen, numAgents = filename.split(".txt")[0].split(".scen") # remove .txt, e.g. empty_8_8-random-9, 32
            mapname = scen.split("-")[0] + ".map" # e.g. empty_8_8.map
            val = parse_path(f) # get the 2 types of paths: the first being a list of agent locations for each timestep, the 2nd being a map for each timestep with -1 if no agent, agent number otherwise
            # print(mapname, bdname, seed, np.count_nonzero(val2 != -1)) # debug statement
            # print("___________________________\n")
            # if idx in valFiles:
            #     res2[mapname + "," + bdname] = val
            # else:
            res1[f"{mapname},{scen},{numAgents}"] = val
            idx += 1
        else:
            raise RuntimeError("bad path dir")
    
    return res1



def main():
    # cmdline argument parsing: take in dirs for paths, maps, and bds, and where you want the outputted npz
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathsIn", help="directory containing txt files of agents and paths taken", type=str)
    parser.add_argument("--pathOutFile", help="output filepath npz to save path", type=str)
    parser.add_argument("--bdIn", help="directory containing txt files with backward djikstra output", type=str)
    parser.add_argument("--bdOutFile", help="output filepath npz to save backward djikstras", type=str)
    parser.add_argument("--mapIn", help="directory containing txt files with obstacles", type=str)
    parser.add_argument("--mapOutFile", help="output filepath npz to save map", type=str)
    # npzMsg = "output file with maps, bds as name->array dicts, along with (mapname, bdname, path) triplets for each EECBS run"
    parser.add_argument("--num_parallel", help="num_parallel", type=int, default=1)
    

    args = parser.parse_args()

    # pathsIn = args.pathsIn
    # bdIn = args.bdIn
    # mapIn = args.mapIn
    # trainOut = args.trainOut

    # instantiate global variables that will keep track of each map and bd that you've encountered
    # maps = {} # maps mapname->np array containing the obstacles in map
    # bds = {} # maps bdname->np array containing bd for each agent in the instance (NOTE: keep track of number agents in bdname)

    # parse each map, add to global dict
    ct = CustomTimer()
    
    assert(args.mapOutFile.endswith(".npz"))
    if os.path.exists(args.mapOutFile):
        print("Map file already exists, skipping map parsing")
        pass
    else:
        with ct("Parsing maps"):
            maps = batch_map(args.mapIn, args.num_parallel) # maps mapname->np array containing the obstacles in map
        np.savez_compressed(args.mapOutFile, **maps)
        ct.printTimes("Parsing maps")

    # parse each bd, add to global dict
    assert(args.bdOutFile.endswith(".npz"))
    if os.path.exists(args.bdOutFile):
        print("BD file already exists, skipping bd parsing")
        pass
    else:
        with ct("Parsing bds"):
            bds = batch_bd(args.bdIn, args.num_parallel)
        np.savez_compressed(args.bdOutFile, **bds)
        ct.printTimes("Parsing bds")

    # parse each path, add to global list
    assert(args.pathOutFile.endswith(".npz"))
    if os.path.exists(args.pathOutFile):
        print(f"Path file {args.pathOutFile} already exists, skipping path parsing")
    else:
        if not os.listdir(args.pathsIn):
            print("WARNING: pathsIn folder is empty. This might be okay if restarting previous run or all failed.")
            return
        with ct("Parsing paths"):
            paths_data = batch_path(args.pathsIn)
        ct.printTimes("Parsing paths")
        with ct("Saving npz"):
            np.savez_compressed(args.pathOutFile, **paths_data)
        ct.printTimes("Saving npz")

    # Verify that the dataloader works by sampling 10 random items
    with ct("Testing dataloader"):
        loader = PipelineDataset(args.mapOutFile, args.bdOutFile, args.pathOutFile, 4, float('inf'), 300, 'current')
        print(len(loader), " train size")
        random_samples = np.random.choice(len(loader), 10)
        for i in random_samples:
            locs, labels, bd, grid = loader[i] # This will fail if not parsed/loaded properly
            assert(labels.shape[1] == 5)
            assert(locs.shape[1] == 2)
            # grid/bd should be 9,9 for single agent but is full map for graph version


# python -m data_collection.data_manipulator --pathsIn=./data_collection/eecbs/raw_data/final_test8/paths/ 
#       --bdIn=./data_collection/eecbs/raw_data/final_test8/bd/ 
#       --bdOutFile=./data_collection/data/benchmark_data/constant_npzs/final_test8_bds.npz 
#       --mapOutFile=./data_collection/data/benchmark_data/constant_npzs/final_test8_map.npz 
#       --mapIn=./data_collection/data/benchmark_data/maps --trainOut=./data_collection/data/logs/EXP0/labels/raw/train_final_test8_0 
#       --num_parallel=1
# python -m data_collection.data_manipulator --pathsIn=data_collection/data/logs/EXP_Collect_BD/iter0/eecbs_outputs/empty_8_8/paths/ --pathOutFile=data_collection/data/logs/EXP_Collect_BD/iter0/eecbs_npzs/empty_8_8_paths.npz --bdIn=data_collection/data/logs/EXP_Collect_BD/iter0/eecbs_outputs/empty_8_8/bd --bdOutFile=data_collection/data/benchmark_data/constant_npzs2/empty_8_8_bds.npz --mapIn=data_collection/data/benchmark_data/maps --mapOutFile=data_collection/data/benchmark_data/constant_npzs2/empty_8_8_map.npz --num_parallel=1
if __name__ == "__main__":
    main()
