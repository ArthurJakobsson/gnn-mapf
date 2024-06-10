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

up = np.array([-1, 0])
down = np.array([1, 0])
left = np.array([0, -1])
right = np.array([0, 1])
stop = np.array([0, 0])
moves_ordered = np.array([up, left, down, right, stop])


def parse_map(path, mapfile, k):
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
    mapdata = np.pad(mapdata, k, mode="constant", constant_values=1)
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
    def __init__(self, map_files, scen_files, k=4, map_path='../map_files/', scen_path = '../scen_files/', first_time=False):
        self.k = 4
        if not first_time:
            with open('saved_map_dict.pickle', 'rb') as handle:
                self.map_dict = pickle.load(handle)
        else:
            self.map_dict = collections.defaultdict(list)
            self.map_dict['map'] = dict()
            self.map_dict['scen'] = dict()

        for map_name in map_files:
            if map_name in self.map_dict['loaded_maps']:
                continue

            self.map_dict['loaded_maps'].append(map_name)
            just_map_name = map_name[0:-4]
            print(just_map_name)
            self.map_dict['map'][just_map_name] = parse_map(map_path, map_name, self.k)
        for scen_f in scen_files:
            if scen_f in  self.map_dict['loaded_scenes']:
                continue

            self.map_dict['loaded_scenes'].append(scen_f)
            scen_name = parse_scen_name(scen_f)
            if not scen_name in self.map_dict['scen']:
                self.map_dict['scen'][scen_name]=collections.defaultdict(list)
            start_loc, goal_loc = parse_scene(scen_path, scen_f)
            # add start and goal
            self.map_dict['scen'][scen_name]['agent_info'].append((start_loc+self.k,goal_loc+self.k))
            # calculate bds
            self.map_dict['scen'][scen_name]['bd'].append(calculate_bds(goal_loc, self.map_dict['map'][scen_name]))
        
        
        with open('saved_map_dict.pickle', 'wb') as handle:
            pickle.dump(self.map_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_map_dict(self):
        return self.map_dict
    

class RunModel():
    def __init__(self, device, model=None, preprocess=None, cs_type="PIBT", m=5, k=4):
        self.m = m
        self.k = k
        self.device = device

        if preprocess==None or model==None:
            raise RuntimeError("No prerpocessing information or model provided")
        map_dict = preprocess.get_map_dict()


        cur_map = map_dict['map']['warehouse-10-20-10-2-2'] #TODO change this to iterate through all maps
        cur_agent_locs = map_dict['scen']['warehouse-10-20-10-2-2']['agent_info'][0][0] #first zero is scen, #second is the start_locs
        cur_bd = map_dict['scen']['warehouse-10-20-10-2-2']['bd'][0]
        cur_data = create_data_object(pos_list=cur_agent_locs, bd_list=cur_bd, grid=cur_map, k=self.k, m=self.m)
        
        cur_data = cur_data.to(self.device)
        # normalize bd, normalize edge attributes
        edge_weights, bd_and_grids = cur_data.edge_attr, cur_data.x
        cur_data.edge_attr = apply_edge_normalization(edge_weights)
        cur_data.x = apply_bd_normalization(bd_and_grids, self.k, self.device)
        _, predictions = model(cur_data)
        # print(torch.argmax(predictions,dim=1))
        logits = torch.exp(predictions)
        # logits = logits / logits.sum()
        
        # Random action #TODO implement O_tie
        action_preferences = torch.multinomial(logits, num_samples=5, replacement=False)

        if cs_type=="PIBT":
            raise NotImplementedError
        else:
            new_agent_locs = self.cs_naive(cur_map, cur_agent_locs, action_preferences)
            pdb.set_trace()

    # CS_naive (pretty inefficient, never ends for large numbers of collisions)
    def cs_naive(self, cur_map, cur_agent_locs, action_preferences):
        collisions = True
        chosen_action = action_preferences[:,0]
        modified_map = cur_map.copy()
        collision_count_iterations = 0 
        pdb.set_trace()
        while (collisions):
            collision_count_iterations += 1
            occupied_nodes = []
            occupied_edges = []
            new_locations = []
            for idx, act in enumerate(chosen_action):
                cur_loc = cur_agent_locs[idx]
                next_loc = cur_loc + moves_ordered[act]
                # Skip if would leave map bounds
                if next_loc[0] < 0 or next_loc[0] >= modified_map.shape[0] or next_loc[1] < 0 or next_loc[1] >= modified_map.shape[1]:
                    new_locations.append(cur_loc) 
                    continue

                # Skip if obstacle
                if modified_map[next_loc[0], next_loc[1]]==1:
                    new_locations.append(cur_loc)
                    continue
                    
                new_locations.append(next_loc)
                occupied_nodes.append(tuple(next_loc))
                occupied_edges.append(tuple([*cur_loc, *next_loc]))
            any_collisions = False
            for idx, (loc, edge) in enumerate(zip(occupied_nodes, occupied_edges)):
                first_loc, second_loc = edge[0:2], edge[2:4]
                cur_loc = cur_agent_locs[idx]
                if(occupied_nodes.count(loc)>1 or tuple([*second_loc, *first_loc]) in occupied_edges):
                    any_collisions = True
                    modified_map[cur_loc[0], cur_loc[1]] = 1 #any agents that have a conflict get turned into an obstacle at current position
                    chosen_action[idx] = 4 #any agent action that results in collision is changed to a stop

            if (not any_collisions):
                print("end")
                collisions = False # this line doesn't actually do anything
                return new_locations
            if (collision_count_iterations>100):
                return cur_agent_locs





if __name__ == "__main__":
    run_name = 'full_sum_aggr'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('../../gnn/model_log/'+run_name+'/max_double_test_acc.pt')
    model.to(device)
    model.eval()

    map_files = ['warehouse-10-20-10-2-2.map']
    scen_files = ['warehouse-10-20-10-2-2-random-1.scen','warehouse-10-20-10-2-2-random-2.scen', 'warehouse-10-20-10-2-2-random-3.scen']

    torch.manual_seed(1072)

    k = 4
    m = 5
    startup = Preprocess(map_files, scen_files, k=k, first_time=False)
    # print(startup.get_map_dict())

    model_run = RunModel(device, model=model, preprocess=startup, cs_type="Naive", k=k, m=m)
    




