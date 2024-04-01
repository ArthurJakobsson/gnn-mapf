import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import pdb
from tqdm import tqdm
import os
import collections

import numpy as np
import torch.optim as optim
import sys

sys.path.insert(0, '/home/arthur/snap/snapd-desktop-integration/current/Documents/gnn-mapf/gnn/')
from dataloader import *
from trainer import *

import matplotlib.pyplot as plt



def parse_map(path, mapfile):
    '''
    takes in a mapfile and returns a parsed np array
    '''
    with open(path+mapfile) as f:
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

def linearize_coordinate(row, col, num_of_cols):
    return num_of_cols * row + col

def parse_scene(scen_path, scene_name):
    start_locations = []
    goal_locations = []

    with open(scen_path+scene_name) as f:
        line = f.readline().strip()
        # if line[0] == 'v':  # Nathan's benchmark
        start_locations = np.array([])
        goal_locations = np.array([])
        sep = "\t"
        for line in f:
            line = line.rstrip() #remove the new line
            tokens = line.split(sep)
            num_of_cols = int(tokens[2])
            num_of_rows = int(tokens[3])
            tokens = tokens[4:]  # Skip the first four elements
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations = np.concatenate((start_locations,np.array([col,row])))
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations = np.concatenate((goal_locations,np.array([col,row])))
    return (start_locations, goal_locations)


def parse_scen_name(scen_name):
    index = scen_name.rindex('random')
    return scen_name[0:index-1]

class Preprocess():
    def __init__(self, map_files, scen_files, map_path='../map_files/', scen_path = '../scen_files/'):
        self.map_dict = dict()
        self.map_dict['map'] = collections.defaultdict(list)
        self.map_dict['scen'] = collections.defaultdict(list)
        for map_name in map_files:
           just_map_name = map_name[0:-4]
           print(just_map_name)
           self.map_dict['map'][just_map_name] = parse_map(map_path, map_name)
        for scen_name in scen_files:
            just_scen_name = parse_scen_name(scen_name)
            self.map_dict['scen'][just_scen_name].append(parse_scene(scen_path, scen_name))

    def get_map_dict(self):
        return self.map_dict

    def get_start_graph(self):






if __name__ == "__main__":
    run_name = 'full_sum_aggr'
    model = torch.load('../../gnn/model_log/'+run_name+'/max_double_test_acc.pt')
    model.eval()

    map_files = ['warehouse-10-20-10-2-2.map']
    scen_files = ['warehouse-10-20-10-2-2-random-1.scen','warehouse-10-20-10-2-2-random-2.scen']

    startup = Preprocess(map_files, scen_files)
    print(startup.get_map_dict())



