import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import argparse



import pdb
from tqdm import tqdm
import os
import collections

import numpy as np
import torch.optim as optim
import sys

from bd_planner import *

# sys.path.insert(0, '/home/arthur/snap/snapd-desktop-integration/current/Documents/gnn-mapf/gnn/')
from dataloader import *
from trainer import *

# tools
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt
import pickle

up = [-1, 0]
down = [1, 0]
left = [0, -1]
right = [0, 1]
stop = [0, 0]
possible_moves = torch.tensor([up, left, down, right, stop])

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

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

# def create_scen_folder(idx):
#     if not os.path.exists("created_scens/created_scen_files_"+str(idx)):
#         os.makedirs("created_scens/created_scen_files_"+str(idx))
#         return str("created_scens/created_scen_files_"+str(idx))
#     else: 
#         return create_scen_folder(idx+1)

def write_line(idx, start_loc, goal_loc, file, map_r, map_c, map_name):
    out_str = str(idx)+"\t" + map_name + "\t" + str(map_r) + "\t" + str(map_c)+"\t"+ \
                str(start_loc[0])+"\t"+str(start_loc[1])+"\t"+str(goal_loc[0])+"\t"+str(goal_loc[1])+"\t1072\n"
    
    file.write(out_str)

def save_scen(start_locs, goal_locs, map, map_name, scen_name, idx, scen_folder, k):

    map_r, map_c = map.shape[0]-2*k, map.shape[1]-2*k
    file = open(scen_folder+"/"+ scen_name+"-custom-"+str(idx)+"-.scen", 'w')
    file.write("version "+str(len(start_locs))+"\n")
    start_locs = np.array(start_locs)-k # (n,2)
    row, col = start_locs[:,0], start_locs[:,1]
    start_locs = np.array([col, row]).T # (n,2)
    
    
    goal_locs = goal_locs-k # (n,2)
    row, col = goal_locs[:,0], goal_locs[:,1]
    goal_locs = np.array([col, row]).T # (n,2)
    # TODO pass map name
    for idx, package in enumerate(zip(start_locs, goal_locs, repeat(file), repeat(map_r), repeat(map_c), repeat(map_name))):
        write_line(idx, *package)
    file.close()

class Preprocess():
    def __init__(self, map_files, scen_files, k=4, map_path='../data_collection/data/benchmark_data/maps', scen_path = '../data_collection/data/benchmark_data/scens', first_time=False):
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
            self.map_dict['scen'][scen_name]['scen_full_name'].append(scen_f[:-5])
            self.map_dict['scen'][scen_name]['agent_info'].append((start_loc+self.k,goal_loc+self.k))
            # calculate bds
            self.map_dict['scen'][scen_name]['bd'].append(calculate_bds(goal_loc, self.map_dict['map'][scen_name]))
        
        
        with open('saved_map_dict.pickle', 'wb') as handle: # TODO make this scen dependent if there is too much data for pickle
            pickle.dump(self.map_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_map_dict(self):
        return self.map_dict
    

class RunModel():
    def __init__(self, device, scen_folder, model=None, preprocess=None, cs_type="PIBT", m=5, k=4, num_agents=100): #TODO change to both 5, 50 and 100
        self.m = m
        self.k = k
        self.device = device
        self.scen_number = 0
        self.scen_folder = scen_folder
        self.model = model
        self.map_dict = preprocess.get_map_dict()
        self.cs_type = cs_type
        self.num_agents = num_agents

        if preprocess==None or model==None:
            raise RuntimeError("No prerpocessing information or model provided")

        self.run_simulation()


    def run_simulation(self):
        for map_name in self.map_dict['map'].keys():
            for scen_idx in range(len(self.map_dict['scen'][map_name]['agent_info'])):
                self.simulate_agent_iterations(map_name, scen_idx)

    def simulate_agent_iterations(self, map_name, scen_idx):
        cur_map = self.map_dict['map'][map_name] 
        scen_full_name = self.map_dict['scen'][map_name]['scen_full_name'][scen_idx]
        cur_agent_locs = self.map_dict['scen'][map_name]['agent_info'][scen_idx][0] #first zero is scen, #second is the start_locs
        cur_agent_goals = self.map_dict['scen'][map_name]['agent_info'][scen_idx][1] #first zero is scen, #second is the start_locs
        cur_agent_locs, cur_agent_goals = cur_agent_locs[0:self.num_agents], cur_agent_goals[0:self.num_agents] # prune agents
        cur_bd = self.map_dict['scen'][map_name]['bd'][scen_idx]
        iteration, solved = 0, False
        while (iteration < 200 and not solved):
            cur_data = create_data_object(pos_list=cur_agent_locs, bd_list=cur_bd, grid=cur_map, k=self.k, m=self.m) 
            cur_data = cur_data.to(self.device)
            
            # normalize bd, normalize edge attributes
            edge_weights, bd_and_grids = cur_data.edge_attr, cur_data.x
            cur_data.edge_attr = apply_edge_normalization(edge_weights)
            cur_data.x = apply_bd_normalization(bd_and_grids, self.k, self.device)
            _, predictions = model(cur_data)
            probabilities = torch.softmax(predictions, dim=1) 
            
            # Random action #TODO implement O_tie - Rishi said we don't want this actually
            if self.cs_type=="PIBT":
                new_agent_locs = self.cs_pibt(self.device, cur_map, cur_agent_locs, probabilities)
            else:
                new_agent_locs = self.cs_naive(self.device, cur_map, cur_agent_locs, probabilities)

            
            save_scen(new_agent_locs, cur_agent_goals, cur_map, map_name, scen_full_name, self.scen_number, self.scen_folder, k)
            self.scen_number+=1

            if (np.all(new_agent_locs==cur_agent_goals)):
                solved = True
                print("Solved")
            cur_agent_locs = new_agent_locs
            iteration+=1

    def cs_pibt(self,device, cur_map, cur_agent_locs, probabilities):
        planned_agents, occupied_nodes, occupied_edges = [], [], []
        action_preferences = torch.multinomial(probabilities, num_samples=5, replacement=False).cpu()
        cur_agent_locs = torch.tensor(cur_agent_locs, dtype=torch.int16)
        move_matrix = torch.zeros((cur_agent_locs.shape[0], 2), dtype=torch.int16)
        #TODO potentially sort agent locations by distance to goal, right now: arbitrary
        for agent_id in range(cur_agent_locs.shape[0]):
            if agent_id not in planned_agents:
                self.pibt(agent_id, action_preferences, planned_agents, move_matrix, occupied_nodes, occupied_edges, cur_agent_locs, cur_map)
        return np.array(cur_agent_locs+move_matrix)

    def pibt(self, agent_id, action_preferences, planned_agents, move_matrix, occupied_nodes, occupied_edges, cur_agent_locs, cur_map):
        '''
        Runs PIBT for a single agent
        Args:
            agent_id: agent id
            move_preferences: (N,5) up-down-left-right-wait
        Returns:
            isvalid: whether we can find a feasible solution
        '''
        num_agents = cur_agent_locs.shape[0]
        def findAgentAtLocation(aLoc):
            for a in range(num_agents):
                if a == agent_id:
                    continue
                if tuple(cur_agent_locs[a]) == tuple(aLoc):
                    return a
            return -1
        moves_ordered = possible_moves[action_preferences[agent_id]] # 0 index is first action to take 
        cur_loc = cur_agent_locs[agent_id]

        for move in moves_ordered:
            next_loc = cur_loc + move

            # Skip if would leave map bounds
            if next_loc[0] < 0 or next_loc[0] >= cur_map.shape[0] or next_loc[1] < 0 or next_loc[1] >= cur_map.shape[1]:
                continue
            # Skip if obstacle
            if cur_map[next_loc[0], next_loc[1]]==1:
                continue

            # Skip if vertex occupied by higher agent
            if tuple(next_loc) in occupied_nodes:
                continue
            # Skip if reverse edge occupied by higher agent
            if tuple([*next_loc, *cur_loc]) in occupied_edges:
                continue
            
            move_matrix[agent_id] = move
            planned_agents.append(agent_id)
            occupied_nodes.append(tuple(next_loc))
            occupied_edges.append(tuple([*cur_loc, *next_loc]))
            conflicting_agent = findAgentAtLocation(next_loc)
            if conflicting_agent != -1 and conflicting_agent not in planned_agents:   
                is_valid = self.pibt(conflicting_agent, action_preferences, planned_agents, move_matrix, occupied_nodes, occupied_edges, cur_agent_locs, cur_map)
                if is_valid:
                    return True
                else:
                    del planned_agents[-1]
                    del occupied_nodes[-1]
                    del occupied_edges[-1]
                    continue
            else:
                return True
            
        return False




    # CS_naive (pretty inefficient, never ends for large numbers of collisions)
    def cs_naive(self, cur_map, cur_agent_locs, probabilities):
        action_preferences = torch.multinomial(probabilities, num_samples=5, replacement=False)
        collisions = True
        chosen_action = action_preferences[:,0]
        modified_map = cur_map.copy()
        collision_count_iterations = 0 
        while (collisions):
            collision_count_iterations += 1
            occupied_nodes = []
            occupied_edges = []
            new_locations = []
            for idx, act in enumerate(chosen_action):
                cur_loc = cur_agent_locs[idx]
                next_loc = cur_loc + possible_moves[act]
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
                print("collision checking finished")
                collisions = False # this line doesn't actually do anything
                return new_locations
            if (collision_count_iterations>500):
                return cur_agent_locs





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", help="experiment folder", type=str)
    parser.add_argument('--firstIter', dest='firstIter', type=lambda x: bool(str2bool(x)))
    parser.add_argument("--source_maps_scens", help="which map+scen folder", type=str)
    parser.add_argument("--iternum", help="iternum", type=int)
    args = parser.parse_args()
    exp_folder, firstIter, source_maps_scens, iternum= args.exp_folder, args.firstIter, args.source_maps_scens, args.iternum

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = torch.load(exp_folder+f"/iter{iternum}/models/max_double_test_acc.pt")
    model.to(device)
    model.eval()

    scen_folder = exp_folder+f"/iter{iternum}/encountered_scens/"
    os.mkdir(scen_folder)

    # map_files = ['warehouse_10_20_10_2_2.map']
    # scen_files = ['warehouse_10_20_10_2_2-random-1.scen','warehouse_10_20_10_2_2-random-2.scen', 'warehouse_10_20_10_2_2-random-3.scen']

    torch.manual_seed(1072)

    # scen_folder = create_scen_folder(0)

    k = 4
    m = 5
    num_agents = 100
    map_path_chosen = source_maps_scens+"/maps/"
    scen_path_chosen = source_maps_scens + "/scens/"

    map_files = os.listdir(map_path_chosen)
    scen_files = os.listdir(scen_path_chosen)
    startup = Preprocess(map_files, scen_files, k=k, map_path=map_path_chosen, scen_path=scen_path_chosen, first_time=firstIter)
    # print(startup.get_map_dict())

    model_run = RunModel(device, scen_folder, model=model, preprocess=startup, cs_type="PIBT", k=k, m=m, num_agents=num_agents)
    




