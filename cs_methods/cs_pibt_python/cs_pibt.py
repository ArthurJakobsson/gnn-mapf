import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import pdb
from tqdm import tqdm
import os

import numpy as np
import torch.optim as optim
import sys

import dataloader
import matplotlib.pyplot as plt


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




if __name__ == "__main__":
  run_name = '20240330-144357'
  model = torch.load('../gnn/'+run_name+'/max_double_test_acc.pt')
  model.eval()
for name, param in model.named_parameters():
  if param.requires_grad:
      print(name, param.data)

