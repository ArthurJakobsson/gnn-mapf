import os.path as osp

import math
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
import pdb

from torch_geometric.data import Data
from tqdm import tqdm
import os
import sys
from multiprocessing import Pool
import pickle

# print(os.path.abspath(os.getcwd()))
sys.path.insert(0, './data_collection/')
import data_manipulator

def slice_maps(pos, curmap, k):
    """
    Turns a full size bd into a 2k+1, 2k+1
    param: pos, bd (current position and bd)
    output: sliced bd
    """
    r,c = pos[0], pos[1]
    return curmap[r-k:r+k+1,c-k:c+k+1] 

def get_neighbors(m, pos, pos_list, k):
    """
    Gets the m closest neighbors *within* 2k+1 bounding
    param: m, pos, pos_list
    output: list of indices of 5 closest
    """
    closest_5_in_box = []

    distances = cdist([pos], pos_list, 'euclidean')[0]
    zipped_lists = zip(distances, pos_list, list(range(0,len(pos_list)))) #knit distances, pos, and idxs
    sorted_lists = sorted(zipped_lists, key = lambda t: t[0])
    count, i = 0,0
    while count<m and i<len(pos_list):
        xdif = pos_list[i][0]-pos[0]
        ydif = pos_list[i][1]-pos[1]
        if abs(xdif) <= k and abs(ydif)<=k:
            count+=1
            closest_5_in_box.append(sorted_lists[i][2])
            # add to list if within bounding box
        i+=1
    return closest_5_in_box

def convert_neighbors_to_edge_list(num_agents, closest_neighbors_idx):            
    # Create arrays to hold source and destination indices and edge_indices
    edge_index, source_indices, dest_indices = [], [], []
    # Iterate through agents and their closest neighbors
    for i in range(num_agents):
        neighbors = closest_neighbors_idx[i]
        source_indices.extend([i] * len(neighbors))
        dest_indices.extend(neighbors)
    # Convert to NumPy arrays
    source_indices = np.array(source_indices)
    dest_indices = np.array(dest_indices)
    # Stack and transpose to create the final edge index
    edge_index = np.vstack((source_indices, dest_indices)).T
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous() #transpose

def get_edge_attributes(edge_index, pos_list):
    # Convert pos_list and edge_index to NumPy arrays
    pos_array = np.array(pos_list)
    edge_index_array = np.array(edge_index)

    # Get the source and destination indices
    source_indices, dest_indices = edge_index_array[0], edge_index_array[1]

    # Get the positions of the sources and destinations
    source_positions, dest_positions = pos_array[source_indices], pos_array[dest_indices]

    # Compute the differences
    differences = source_positions - dest_positions

    return differences

def get_idx_name(raw_path):
    if 'val' in raw_path:
        return 'val'
    elif 'train' in raw_path:
        return 'train'
    else:
        return 'unknownNPZ'

def apply_masks(data_len, curdata):
    tr_mask, te_mask = np.zeros(data_len), np.zeros(data_len)
    tr_mask[:int((3/4)*data_len)] = 1
    te_mask[int((3/4)*data_len):] = 1
    curdata.train_mask = torch.tensor(tr_mask, dtype=torch.bool)
    curdata.test_mask = torch.tensor(te_mask, dtype=torch.bool)
    return curdata

def apply_edge_normalization(edge_weights):
    # TODO have an option to choose method based on flags
    edge_weights = torch.exp(-edge_weights)
    return edge_weights

def apply_bd_normalization(bd_grid, k, device):
    center = bd_grid[:, 1, k, k].unsqueeze(1).unsqueeze(2)
    bd_grid[:, 1, :, :] -= center
    bd_grid[:, 1, :, :] *= (1 - bd_grid[:, 0, :, :])
    bd_grid[:, 1, :, :] /= k
    bd_grid[:, 1, :, :] = torch.clamp(bd_grid[:, 1, :, :], min=-10.0, max=10.0)

    return bd_grid

def create_data_object(pos_list, bd_list, grid, k, m, labels=np.array([])):
    num_agents = len(pos_list)
    x = [np.array([slice_maps(pos, grid, k), slice_maps(pos, bd, k)]) for (pos, bd) in zip(pos_list, bd_list)]

    m_closest_nborsIdx_list = [get_neighbors(m, pos, pos_list, k) for pos in pos_list] # (m,n)
    edge_index = convert_neighbors_to_edge_list(num_agents, m_closest_nborsIdx_list) # (2, num_edges)
    edge_attr = get_edge_attributes(edge_index, pos_list) # (num_edges,2 )

    # Tensorify
    x = torch.tensor(np.array(x), dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.int8) # up down left right stay (5 options)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = labels)

class MyOwnDataset(Dataset):
    def __init__(self, root, file_path_loaded, device, iternum, num_cores=20, transform=None, pre_transform=None, pre_filter=None, generate_initial=True):
        self.file_path_loaded = file_path_loaded
        self.num_cores = num_cores
        self.k = 4 # padding size
        self.m = 5 # number of agents considered close
        self.size = float('inf')
        self.max_agents = 500
        self.iternum = iternum
        # self.data_dictionaries = []
        self.length = 0
        self.device = device
        self.generate_initial = generate_initial
        meta = {"length": 0, "ranges": [(0,"nothing")]}
        with open(file_path_loaded["meta_data"], 'wb') as fp:
            pickle.dump(meta, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load_metadata()

    def load_metadata(self):
        if osp.isfile(self.file_path_loaded["meta_data"]):
            with open(self.file_path_loaded["meta_data"], 'rb') as fp:
                meta_data = pickle.load(fp)
                self.length = meta_data["length"]
        else:
            self.length = 0 
        return self.length

    @property
    def raw_file_names(self):
        npz_paths = os.listdir(self.file_path_loaded["eecbs_npzs"])
        return npz_paths

    # @property
    # def processed_file_names(self):
    #     file_names = []
    #     for i in range(self.length):
    #         file_names.append(f"data_{i}.pt")
    #     return  file_names 

    def download():
        raise ImportError
    
    @property
    def num_classes(self) -> int:
        return 5

    def create_and_save_graph(self, idx, time_instance):
        # Graphify
        if not time_instance: 
            return #idk why but the last one is None
        pos_list, labels, bd_list, grid = time_instance # (1), (2,n), (md,md): md=map dim with pad, n=num_agents

        curdata = create_data_object(pos_list, bd_list, grid, self.k, self.m, labels)
        curdata = apply_masks(len(curdata.x), curdata)
        torch.save(curdata, osp.join(self.file_path_loaded["processed"], f"data_{idx}.pt"))

    def process(self):
        # if self.iternum==0 and not self.generate_initial:
        #     return #if pt's are made already skip the first iteration of pt making
        for raw_path in self.raw_paths: #TODO check if new npzs are read
            if "maps" in raw_path or "bds" in raw_path:
                continue
            cur_path_iter = raw_path.split("_")[-1][:-4]
            if not (int(cur_path_iter)==self.iternum):
                continue

            cur_dataset = data_manipulator.PipelineDataset(raw_path, self.k, self.size, self.max_agents)
            print(f"Loading: {raw_path} of size {len(cur_dataset)}")
            idx_list = np.array(range(len(cur_dataset)))

            self.create_and_save_graph(0, cur_dataset[0])
            with Pool(self.num_cores) as p: 
                p.starmap(self.create_and_save_graph, zip(idx_list, cur_dataset))
            # for time_instance in tqdm(cur_dataset):
                
            self.length += len(cur_dataset)

            with open(self.file_path_loaded["meta_data"], 'rb') as file:
                meta_data = pickle.load(file)

            meta_data["length"] = self.length
            meta_data["ranges"].append((self.length, self.file_path_loaded["processed"]))
            # Overwrite the picle file
            with open(self.file_path_loaded["meta_data"], 'wb') as file:
                pickle.dump(meta_data, file)
            # meta = torch.tensor([self.length])
            # torch.save(meta, self.file_path_loaded["meta_data"])

    def len(self):
        return self.length

    def get(self, idx):
        with open(self.file_path_loaded["meta_data"], 'rb') as file:
            meta_data = pickle.load(file)
        ranges = meta_data["ranges"]
        folder_num = 0 
        while idx > ranges[folder_num][0]:
            folder_num+=1
        
        chosen_folder = ranges[folder_num][1]
        file_idx = idx - ranges[folder_num-1][0]
        
        curdata = torch.load(osp.join(chosen_folder, f"data_{file_idx}.pt"))
        curdata = curdata.to(self.device)

        # normalize bd, normalize edge attributes
        edge_weights, bd_and_grids = curdata.edge_attr, curdata.x
        curdata.edge_attr = apply_edge_normalization(edge_weights)
        curdata.x = apply_bd_normalization(bd_and_grids, self.k, self.device)

        return curdata

# john = MyOwnDataset('map_data_big2d_new')
# john.get(5)
