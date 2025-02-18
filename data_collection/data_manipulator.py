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
import ray.util.multiprocessing
# import multiprocessing
# print(os.path.abspath(os.getcwd()))
# sys.path.insert(0, './data_collection/')
# sys.path.append(os.path.abspath(os.getcwd())+"/custom_utils/")
from custom_utils.custom_timer import CustomTimer
from custom_utils.common_helper import getMapScenAgents

'''
0. parse bd (fix eecbs by having it make another txt output, parse it here)
    note that each agent has its own bd for a given map
1. make it a batch that accepts directories instead of files
2. dictionary that maps map_name to map
3. dictionary that maps bd_name to bd
4. write tuples of (map name, bd name, paths np) to npz
'''
 
# mapsToMaxNumAgents = {
#     "Berlin_1_256": 1000,
#     "Boston_0_256": 1000,
#     "Paris_1_256": 1000,
#     "brc202d": 1000,
#     "den312d": 1000, 
#     "den520d": 1000,
#     "dense_map_15_15_0":50,
#     "dense_map_15_15_1":50,
#     "corridor_30_30_0":50,
#     "empty_8_8": 32,
#     "empty_16_16": 128,
#     "empty_32_32": 512,
#     "empty_48_48": 1000,
#     "ht_chantry": 1000,
#     "ht_mansion_n": 1000,
#     "lak303d": 1000,
#     "lt_gallowstemplar_n": 1000,
#     "maze_128_128_1": 1000,
#     "maze_128_128_10": 1000,
#     "maze_128_128_2": 1000,
#     "maze_32_32_2": 333,
#     "maze_32_32_4": 395,
#     "orz900d": 1000,
#     "ost003d": 1000,
#     "random_32_32_10": 461,
#     "random_32_32_10_custom_0": 461,
#     "random_32_32_10_custom_1": 461,
#     "random_32_32_20": 409,
#     "random_64_64_10_custom_0": 1000,
#     "random_64_64_10_custom_1": 1000,
#     "random_64_64_10": 1000,
#     "random_64_64_20": 1000,
#     "room_32_32_4": 341,
#     "room_64_64_16": 1000,
#     "room_64_64_8": 1000,
#     "w_woundedcoast": 1000,
#     "warehouse_10_20_10_2_1": 1000,
#     "warehouse_10_20_10_2_2": 1000,
#     "warehouse_20_40_10_2_1": 1000,
#     "warehouse_20_40_10_2_2": 1000,
# }

class PipelineDataset(Dataset):
    '''
    A dataset loader that allows you to store eecbs instances.
    '''

    # instantiate class variables
    def __init__(self, mapFileNpz, goalsFileNpz, bdFileNpz, pathFileNpz, k, size, max_agents, 
                 num_multi_inputs, num_multi_outputs, helper_bd_preprocess="middle"):
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
        priorities: a dictionary mapping eecbs instance names to the priorities for agents.
        helper_bd_preprocess: method by which we center helper backward dijkstras. can be 'middle', 'current', or 'subtraction'.
        '''
        # raw_folder = numpy_data_path.split("train_")[0]
        # loaded_maps = np.load(raw_folder+"maps.npz")
        # loaded_bds = np.load(numpy_data_path.split(".npz")[0]+"_bds.npz")
        # read in the dataset, saving map, bd, and path info to class variables
        # loaded = np.load(numpy_data_path)
        # self.numpy_data_path = numpy_data_path
        assert(mapFileNpz.endswith(".npz") and goalsFileNpz.endswith(".npz")
               and bdFileNpz.endswith(".npz") and pathFileNpz.endswith(".npz"))
        
        self.maps = dict(np.load(mapFileNpz))

        # self.bds = dict(np.load(bdFileNpz))
        scen_to_goal_indices = dict(np.load(goalsFileNpz)) # scen to goals: dict[scenname] = (n,) goal_indices
        self.goal_to_bd = dict(np.load(bdFileNpz))['arr_0'] # goal_to_bd[goal_index] = bd (w, h)
        self.scen_to_goal_indices = scen_to_goal_indices

        paths = dict(np.load(pathFileNpz)) # Note: Very important to make this a dict() otherwise lazy loading kills performance later on
        self.tn2 = {k[:-6]: v for k, v in paths.items() if k[-6:] == "_paths"}
        self.priorities = {k: v for k, v in paths.items() if k[-11:] == "_priorities"}

        self.k = k
        self.size = size
        # self.parse_npz(loaded) # TODO change len(dataloader) = max_timesteps
        self.max_agents = max_agents

        self.num_multi_inputs = num_multi_inputs
        self.num_multi_outputs = num_multi_outputs

        self.helper_bd_preprocess = helper_bd_preprocess

        self.parse_npz2()


    # get number of instances in total (length of training data)
    def __len__(self):
        return self.length # go through the tn2 dict with # data and np arrays saved, and sum all the ints

    # # return the data for a particular instance: the location, bd, and map
    # def __getitem__(self, idx):
    #     '''
    #     INPUT: index (must be smaller than len(self))
    #     OUTPUT: map, bd, and direction
    #         map: (2k+1, 2k+1)
    #         bd: (2k+1, 2k+1)
    #         other agent bds: (4,2k+1,2k+1)
    #         direction: (2)
    #     centered version. when passing in the map and bd, return a (2k+1,2k+1) window centered at current location of agent.
    #     '''
    #     if idx >= self.__len__():
    #         print("Index too large for {}-sample dataset".format(self.__len__()))
    #         return
    #     bd, grid, paths, timestep, max_timesteps, priorities = self.find_instance(idx)
    #     cur_locs = paths[timestep] # (N,2)
    #     next_locs = paths[timestep+1] if timestep+1 < max_timesteps else cur_locs # (N,2)
    #     end_locs = paths[-1]
    #     deltas = next_locs - cur_locs # (N,2)

    #     # Define the mapping from direction vectors to indices
    #     direction_labels = np.array([(0,0), (0,1), (1,0), (-1,0), (0,-1)]) # (5,2)
    #     # Find the index of each direction in the possible_directions array
    #     indices = np.argmax(np.all(deltas[:, None] == direction_labels, axis=2), axis=1)
    #     # Create a one-hot encoded array using np.eye
    #     labels = np.eye(direction_labels.shape[0])[indices]
    #     # assert(np.all(labels == slow_labels))
    #     return cur_locs, labels, bd, grid, end_locs, priorities

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
        bd, grid, paths, timestep, max_timesteps, priorities = self.find_instance(idx)
        # cur_locs = paths[timestep] # (N,2) 

        start_idle_steps = max(0, self.num_multi_inputs-timestep)
        input_move_steps = self.num_multi_inputs-start_idle_steps
        # print(f"multi_input_locs: {start_idle_steps}*[0] + [{timestep-input_move_steps}: {timestep})]")
        input_locs = np.vstack((np.tile(paths[0], (start_idle_steps, 1, 1)),
                                     paths[timestep-input_move_steps: timestep])) # (N,h,2), h history (input steps)
        assert(input_locs.shape[0] == self.num_multi_inputs)

        # next_locs = paths[timestep+1] if timestep+1 < max_timesteps else cur_locs # (N,2)
        # deltas = next_locs - cur_locs # (N,2)

        output_move_steps = min(max_timesteps - timestep - 1, self.num_multi_outputs)
        end_idle_steps = self.num_multi_outputs - output_move_steps
        # print(f"cur_multi_output_locs: [{timestep}: {timestep + output_move_steps}) + {end_idle_steps}*[{timestep + output_move_steps - 1}]")
        # print(f"next_multi_output_locs: [{timestep+1}: {timestep + output_move_steps+1}) + {end_idle_steps}*[{timestep + output_move_steps}]")
        cur_multi_output_locs = np.vstack((paths[timestep: timestep + output_move_steps],
                                   np.tile(paths[timestep + output_move_steps - 1], (end_idle_steps, 1, 1))))
        next_multi_output_locs = np.vstack((paths[timestep+1: timestep + output_move_steps+1],
                                   np.tile(paths[timestep + output_move_steps], (end_idle_steps, 1, 1))))
        assert(cur_multi_output_locs.shape[0] == next_multi_output_locs.shape[0] == self.num_multi_outputs)
        
        deltas = (next_multi_output_locs-cur_multi_output_locs).swapaxes(0, 1) # (N,ns,2), ns = num steps we predict (3 steps: 125 1-hot vector)
        # assert(np.all(np.equal(cur_multi_locs[1:], next_multi_locs[:-1])))

        # Define the mapping from direction vectors to indices
        direction_labels = np.array([(0,0), (0,1), (1,0), (-1,0), (0,-1)]) # (5,2)
        # Find the index of each direction in the possible_directions array
        
        indices = np.argmax(np.all(deltas[:, :, None] == direction_labels, axis=3), axis=2)
        # assert(np.all(np.equal(indices, indices2[:,0])))
        weights = [5**i for i in range(self.num_multi_outputs)]
        one_hot_indices = np.dot(indices, weights)
        
        # Create a one-hot encoded array using np.eye
        labels = np.eye(direction_labels.shape[0]**self.num_multi_outputs)[one_hot_indices]
        
        output_locs = paths[-1]
        # assert(np.all(labels == slow_labels))
        # return cur_locs, labels, bd, grid, end_locs, priorities
        return input_locs, labels, bd, grid, output_locs, priorities

    def find_instance(self, idx): 
        '''
        returns the backward dijkstra, map, path arrays, and indices to get into the path array
        '''

        # pdb.set_trace()
        assert(idx < self.length)
        # items = list(self.tn2.items())
        total_sum = 0
        key_to_use = None
        timestep_to_use = 0
        for aKey, pathVec in self.tn2.items(): # pathVec is (T,N,2)
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
        # mapname, scenname, num_agents = key_to_use.split(",")
        # num_agents = int(num_agents)
        mapname, scenname, custom_scenname = key_to_use.split(",")
        # num_agents = int(num_agents)
        grid = self.maps[mapname]  # (H,W)
        paths = self.tn2[key_to_use] + self.k # (T,N,2)
        max_timesteps = paths.shape[0]
        num_agents = paths.shape[1]
        
        # bd = self.bds[scenname][:num_agents] # (N,H,W)
        bd = self.goal_to_bd[self.scen_to_goal_indices[scenname]] 
        assert(bd.shape[0] >= num_agents)
        # pdb.set_trace()
        priorities = self.priorities[key_to_use+"_priorities"] # (N,)

        return bd, grid, paths, timestep_to_use, max_timesteps, priorities

    # def parse_npz(self, loaded_paths, loaded_maps, loaded_bds):
    #     self.tn2 = {k:v for k, v in loaded_paths.items()}
    #     self.maps = {k:v for k, v in loaded_maps.items()}
    #     self.bds = {k:v for k, v in loaded_bds.items()}

    #     totalT = 0 
    #     for ky, v in self.tn2.items():
    #         t, n, _ = np.shape(v)
    #         self.tn2[ky] = (t*n, v)
    #         totalT += t
    #     self.length = totalT # number of paths = number of timesteps
    #     # self.twh = dict(items[k:]) # get all the paths in (t,w,h) form
    #     npads = ((0,0),(self.k, self.k), (self.k, self.k))
    #     for key in self.bds:
    #         self.bds[key] = np.pad(self.bds[key], npads, mode="constant", constant_values=1073741823)
    #         # self.bds[key] = np.transpose(self.bds[key], (0, 2, 1)) # (n,h,w) -> (n,w,h) NOTE that originally all bds are parsed in transpose TODO did i fix this correctly
    #     for key in self.maps:
    #         self.maps[key] = np.pad(self.maps[key], self.k, mode="constant", constant_values=1)

    def parse_npz2(self):
        totalT = 0 
        to_remove = []

        for ky, v in self.tn2.items():
            kyname = ky.split(",")[1] # scenname
            if any(scenname == kyname for scenname in self.scen_to_goal_indices.keys()):
                t, n, _ = np.shape(v)
                totalT += t
            else:
                to_remove.append(ky)

        for ky in to_remove:
            self.tn2.pop(ky, None)

        self.length = totalT

        # npads = ((0,0),(self.k, self.k), (self.k, self.k))
        npads = ((self.k, self.k), (self.k, self.k))

        goal_to_bd_padded = []
        for goal in range(len(self.goal_to_bd)):
            goal_to_bd_padded.append(np.pad(self.goal_to_bd[goal], npads, mode="constant", constant_values=1073741823))
        self.goal_to_bd = np.asarray(goal_to_bd_padded)

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
    w = h = 0
    # priorities
    priorities = None
    # maps timesteps to a list of agent coordinates
    timestepsToMaps = defaultdict(list)
    # get max number of timesteps by counting number of commas
    maxTimesteps = 0
    with open(pathfile, 'r') as fd:
        linenum = 0
        for line in fd.readlines():
            if linenum == 0 or linenum == 1:
                linenum += 1
                continue # ignore dimension line and priorities line
            timesteps = 0
            for c in line:
                if c == ',': timesteps += 1
            maxTimesteps = max(maxTimesteps, timesteps)
            linenum += 1

    # get path for each agent and update dictionary of maps accordingly
    with open(pathfile, 'r') as fd:
        linenum = 0
        for linenum, line in enumerate(fd.readlines()):
            if linenum == 0: # parse dimensions of map and keep going
                line = line.strip().split(",")
                w = int(line[0])
                h = int(line[1])
                continue
            elif linenum == 1: # parse priorities and keep going
                priorities_str = line.strip().split()[1] # comma-separated string
                priorities = list(map(int, priorities_str.split(",")[:-1]))
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
                row = int(temp[0][1:])
                col = int(temp[1][:-1])
                # if you're at the last coordinate then append it to the rest of the maps
                if i == len(rawCoords) - 1:
                    while i != maxTimesteps:
                        timestepsToMaps[i].append([row, col])
                        i += 1
                else: timestepsToMaps[i].append([row, col])

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
    return res, np.asarray(priorities)

def parse_bd(bdfile):
    '''
    parses a txt file of bd info for each agent
    input: bdfile (string)
    output: (N,H,W)
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
    
    with ray.multiprocessing.Pool(processes=num_parallel) as pool:
        results = pool.starmap(parse_map, inputs_list)

    for i in range(len(inputs_list)):
        filename = filenames_list[i]
        res[filename] = results[i]
        
    return res

# def batch_bd(dir, num_parallel):
#     '''
#     goes through a directory of bd outputs, parsing each one and saving to a dictionary
#     input: directory of backward djikstras (string)
#     output: dictionary mapping filenames to backward djikstras
#     '''
#     # assert(1 + 1 == 3)
#     res = {} # string->np
#     inputs_list = []
#     filenames_list = []
#     # iterate over files in directory, parsing each map
#     for filename in os.listdir(dir):
#         f = os.path.join(dir, filename)
#         # checking if it is a file
#         if os.path.isfile(f):
#             scenname, agents = filename.split(".")[:2] # e.g. Paris_1_256-random-1.10.txt, where 1 is scen, 10 is agents
#             if num_parallel == 1:
#                 res[scenname] = parse_bd(f)
#             else:
#                 inputs_list.append((f,))  # Note, need to pass in as tuple for use with starmap
#                 filenames_list.append(scenname)
#         else:
#             raise RuntimeError("bad bd dir")
    
#     if num_parallel == 1:
#         return res

#     with multiprocessing.Pool(processes=num_parallel) as pool:
#         results = pool.starmap(parse_bd, inputs_list)

#     for i in range(len(inputs_list)):
#         filename = filenames_list[i]
#         res[filename] = results[i]
#     return res


def batch_bd(bdInDir, scenInDir, num_parallel):
    '''
    goes through a directory of bd outputs, parsing each one and saving to a dictionary
    input: directory of backward djikstras (string)
    output: dictionary mapping goals to backward djikstras
    '''
    goal_to_index = {} # map goal tuple to index in goal_to_bd: tuple->int
    scen_to_goals = {} # scen to array of goal indexes per agent: str->np (n,)
    goal_bds = [] # [np (w, h)]
    inputs_list = [] # scen_bds filenames
    scennames_list = []

    def add_unique_goals_to_dict(agent_to_bd, scenname):
        with open(os.path.join(scenInDir, scenname) + ".scen", "r") as scen_file:
            goals = [' '.join(line.strip().split()[-3:-1]) for line in scen_file.readlines()[1:]]

        for agent, bd in enumerate(agent_to_bd):
            goal = goals[agent]
            if goal not in goal_to_index:
                index = len(goal_to_index)
                goal_to_index[goal] = index
                goal_bds.append(bd) # add to list of unique goals
        scen_to_goals[scenname] = np.asarray([goal_to_index[goal] for goal in goals])

    for filename in os.listdir(bdInDir):
        f = os.path.join(bdInDir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            scenname, _ = filename.split(".")[:2] # e.g. Paris_1_256-random-1.10.txt, where 1 is scen, 10 is agents
            if num_parallel == 1:
                agent_to_bd = parse_bd(f) # list
                add_unique_goals_to_dict(agent_to_bd, scenname)
            else:
                inputs_list.append((f,))  # Note, need to pass in as tuple for use with starmap
                scennames_list.append(scenname)
        else:
            raise RuntimeError("bad bd dir")

    if num_parallel == 1:
        print("# max goal index:", max([max(scen_to_goals[scenname]) for scenname in scen_to_goals]))
        print("# unique BDs:", np.asarray(goal_bds).shape)
        return scen_to_goals, np.asarray(goal_bds)

    with ray.util.multiprocessing.Pool(processes=num_parallel) as pool:
        results = pool.starmap(parse_bd, inputs_list)

    for i in range(len(inputs_list)):
        scenname = scennames_list[i]
        agent_to_bd = results[i] # list
        add_unique_goals_to_dict(agent_to_bd, scenname, goal_bds)

    return scen_to_goals, np.asarray(goal_bds)


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
            # Prior: [SCENNAME].scen[AGENTNUM].[extra].txt   [extra] is used for encountered scens and is optional
            # scen, numAgents = filename.split(".txt")[0].split(".scen") # remove .txt, e.g. empty_8_8-random-9, 32
            # mapname = scen.split("-")[0] + ".map" # e.g. empty_8_8.map
            mapname, scen, numAgents = getMapScenAgents(filename)
            val, priorities = parse_path(f) # get the 2 types of paths: the first being a list of agent locations for each timestep, the 2nd being a map for each timestep with -1 if no agent, agent number otherwise
            # print(mapname, bdname, seed, np.count_nonzero(val2 != -1)) # debug statement
            # print("___________________________\n")
            # if idx in valFiles:
            #     res2[mapname + "," + bdname] = val
            # else:
            res1[f"{mapname}.map,{scen},{numAgents}_paths"] = val
            res1[f"{mapname}.map,{scen},{numAgents}_priorities"] = priorities
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
    parser.add_argument("--goalsOutFile", help="output filepath npz to save indexes of bds for each scen", type=str)
    parser.add_argument("--bdOutFile", help="output filepath npz to save backward djikstras for unique goals", type=str)
    parser.add_argument("--mapIn", help="directory containing txt files with obstacles", type=str)
    parser.add_argument("--scenIn", help="directory containing scen files", type=str)
    parser.add_argument("--mapOutFile", help="output filepath npz to save map", type=str)
    # npzMsg = "output file with maps, bds as name->array dicts, along with (mapname, bdname, path) triplets for each EECBS run"
    parser.add_argument("--num_parallel", help="num_parallel", type=int, default=1)
    

    args = parser.parse_args()

    # instantiate global variables that will keep track of each map and bd that you've encountered
    # maps = {} # maps mapname->np array containing the obstacles in map
    # bds = {} # maps bdname->np array containing bd for each agent in the instance (NOTE: keep track of number agents in bdname)

    # parse each map, add to global dict
    ct = CustomTimer()
    
    # constants_generator
    if args.bdIn and args.goalsOutFile and args.bdOutFile \
                 and args.mapIn and args.scenIn and args.mapOutFile:
        print(f"Running data_manipulator for constants_generator ({args.bdIn})...")
        assert(args.mapOutFile.endswith(".npz"))
        if os.path.exists(args.mapOutFile):
            print("Map file already exists, skipping map parsing")
            pass
        else:
            with ct("Parsing maps"):
                maps = batch_map(args.mapIn, args.num_parallel) # maps mapname->np array containing the obstacles in map
                print("Created all maps")
            os.makedirs(os.path.dirname(args.mapOutFile), exist_ok=True)
            np.savez_compressed(f"{args.mapOutFile}", **maps)
            ct.printTimes("Parsing maps")
        # parse each bd, add to global dict
        assert(args.goalsOutFile.endswith("_goals.npz"))
        os.makedirs(os.path.dirname(args.goalsOutFile), exist_ok=True)
        assert(args.bdOutFile.endswith("_bds.npz"))
        os.makedirs(os.path.dirname(args.bdOutFile), exist_ok=True)

        if os.path.exists(args.bdOutFile) and os.path.exists(args.goalsOutFile):
            print("BD file already exists, skipping bd parsing")
        else:
            with ct("Parsing bds"):
                scen_to_goals, goal_bds = batch_bd(args.bdIn, args.scenIn, args.num_parallel)
            np.savez_compressed(args.goalsOutFile, **scen_to_goals)
            np.savez_compressed(args.bdOutFile, goal_bds)

            ct.printTimes("Parsing bds")

    # eecbs_batchrunner
    if args.pathsIn and args.pathOutFile:
        print("Running data_manipulator for eecbs_batchrunner...")
        # parse each path, add to global list
        assert(args.pathOutFile.endswith(".npz"))
        print(args.pathOutFile)
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
    # with ct("Testing dataloader"):
    #     loader = PipelineDataset(args.mapOutFile, args.bdOutFile, args.pathOutFile, 4, float('inf'), 300, 'current')
    #     print(len(loader), " train size")
    #     random_samples = np.random.choice(len(loader), 10)
    #     for i in random_samples:
    #         locs, labels, bd, grid, goal_locs = loader[i] # This will fail if not parsed/loaded properly
    #         assert(labels.shape[1] == 5)
    #         assert(locs.shape[1] == 2)
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
