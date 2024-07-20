import os
import argparse
import numpy as np
import torch

def parse_scene(scen_file):
    """Input: scenfile
    Output: start_locations, goal_locations
    """
    start_locations = []
    goal_locations = []

    with open(scen_file) as f:
        line = f.readline().strip()
        # if line[0] == 'v':  # Nathan's benchmark
        start_locations = list()
        goal_locations = list()
        sep = "\t"
        for line in f:
            line = line.rstrip() #remove the new line
            tokens = line.split(sep)
            # num_of_cols = int(tokens[2])
            # num_of_rows = int(tokens[3])
            tokens = tokens[4:]  # Skip the first four elements
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row,col))
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row,col))
    return np.array(start_locations), np.array(goal_locations)


def main(args: argparse.ArgumentParser):
    # Load the map
    if not os.path.exists(args.mapNpzFile):
        raise FileNotFoundError('Map file: {} not found.'.format(args.mapNpzFile))
    map_npz = np.load(args.mapNpzFile)
    if args.mapName not in map_npz:
        raise ValueError('Map name not found in the map file.')
    map_grid = map_npz[args.mapName] # (H,W)

    # Load the scen
    if not os.path.exists(args.scenFile):
        raise FileNotFoundError('Scen file: {} not found.'.format(args.scenFile))
    start_locations, goal_locations = parse_scene(args.scenFile) # Each (max agents,2)
    num_agents = args.agentNum
    if start_locations.shape[0] < num_agents:
        raise ValueError('Not enough agents in the scen file.')
    start_locations = start_locations[:num_agents]
    goal_locations = goal_locations[:num_agents]

    # Load the bd
    if not os.path.exists(args.bdNpzFile):
        raise FileNotFoundError('BD file: {} not found.'.format(args.bdNpzFile))
    bd_npz = np.load(args.bdNpzFile)
    bd = bd_npz[args.scenFile] # (max agents,H,W)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Map / scen parameters
    parser.add_argument('--mapNpzFile', type=str, required=True)
    parser.add_argument('--mapName', type=str, required=True)
    parser.add_argument('--scenFile', type=str, required=True)
    parser.add_argument('--agentNum', type=int, required=True)
    parser.add_argument('--bdNpzFile', type=str, required=True)
    # Simulator parameters
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--maxSteps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shieldType', type=str, default='CS-PIBT')
    # Output parameters
    parser.add_argument('--outputCSVFile', type=str, required=True)
    parser.add_argument('--outputPathsFile', type=str, required=True)
    args = parser.parse_args()
    print(args.config)