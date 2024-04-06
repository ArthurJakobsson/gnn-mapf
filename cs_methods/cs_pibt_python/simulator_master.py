import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import pdb
from tqdm import tqdm
import os
import collections

import numpy as np
import torch.optim as optim
import sys

from bd_planner import *

sys.path.insert(0, '/home/arthur/snap/snapd-desktop-integration/current/Documents/gnn-mapf/gnn/')
from dataloader import *
from trainer import *

# tools
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt
import pickle




def parse_map(path, mapfile):
    '''
    takes in a mapfile and returns a parsed np array
    '''
    with open(path+mapfile) as f:
        line = f.readline()  # "type octile"

        line = f.readline()
        height = int(line.split(' ')[1])

        line = f.readline()
        width = int(line.split(' ')[1])

        line = f.readline()  # "map\n"
        assert(line == "map\n")

        mapdata = np.array([list(line.rstrip()) for line in f])

    mapdata[mapdata == '.'] = 0
    mapdata[mapdata == '@'] = 1
    mapdata[mapdata == 'T'] = 1
    mapdata = mapdata.astype(int)
    return mapdata

def parse_scene(scen_path, scene_f):
    start_locations = []
    goal_locations = []

    with open(scen_path+scene_f) as f:
        line = f.readline().strip()
        # if line[0] == 'v':  # Nathan's benchmark
        start_locations = list()
        goal_locations = list()
        sep = "\t"
        for line in f:
            line = line.rstrip() #remove the new line
            tokens = line.split(sep)
            num_of_cols = int(tokens[2])
            num_of_rows = int(tokens[3])
            tokens = tokens[4:]  # Skip the first four elements
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row,col))
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row,col))
    return (np.array(start_locations), np.array(goal_locations))


def parse_scen_name(scen_name):
    index = scen_name.rindex('random')
    return scen_name[0:index-1]

def calculate_bds(goals, map_arr):
    bds = list()
    env = EnvironmentWrapper(map_arr)
    with Pool() as pool:
        for heuristic in pool.starmap(computeHeuristicMap, zip(repeat(env), goals)):
            bds.append(heuristic)
    return np.array(bds)

class Preprocess():
    def __init__(self, map_files, scen_files, map_path='../map_files/', scen_path = '../scen_files/', first_time=False):
        if not first_time:
            with open('saved_map_dict.pickle', 'rb') as handle:
                self.map_dict = pickle.load(handle)
        else:
            self.map_dict = collections.defaultdict(list)
            self.map_dict['map'] = dict()
            self.map_dict['scen'] = dict()

        pdb.set_trace()
        for map_name in map_files:
            if map_name in self.map_dict['loaded_maps']:
                continue

            self.map_dict['loaded_maps'].append(map_name)
            just_map_name = map_name[0:-4]
            print(just_map_name)
            self.map_dict['map'][just_map_name] = parse_map(map_path, map_name)
        for scen_f in scen_files:
            if scen_f in  self.map_dict['loaded_scenes']:
                continue

            self.map_dict['loaded_scenes'].append(scen_f)
            scen_name = parse_scen_name(scen_f)
            if not scen_name in self.map_dict['scen']:
                self.map_dict['scen'][scen_name]=collections.defaultdict(list)
            start_loc, goal_loc = parse_scene(scen_path, scen_f)
            # add start and goal
            self.map_dict['scen'][scen_name]['agent_info'].append((start_loc,goal_loc))
            # calculate bds
            self.map_dict['scen'][scen_name]['bd'].append(calculate_bds(goal_loc, self.map_dict['map'][scen_name]))
        
        
        with open('saved_map_dict.pickle', 'wb') as handle:
            pickle.dump(self.map_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_map_dict(self):
        return self.map_dict


if __name__ == "__main__":
    run_name = 'full_sum_aggr'
    model = torch.load('../../gnn/model_log/'+run_name+'/max_double_test_acc.pt')
    model.eval()

    map_files = ['warehouse-10-20-10-2-2.map']
    scen_files = ['warehouse-10-20-10-2-2-random-1.scen','warehouse-10-20-10-2-2-random-2.scen', 'warehouse-10-20-10-2-2-random-3.scen']

    startup = Preprocess(map_files, scen_files, first_time=False)
    # print(startup.get_map_dict())



