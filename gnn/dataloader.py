import os.path as osp

import math
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
import pdb
import data_manipulator2 as data_manipulator
from torch_geometric.data import Data
from tqdm import tqdm

def slice_maps(pos, curmap, k):
    """
    Turns a full size bd into a 2k+1, 2k+1
    param: pos, bd (current position and bd)
    output: sliced bd
    """
    r,c = pos[0], pos[1]
    return curmap[r-k:r+k+1,c-k:c+k+1] #TODO: adjust +1 or -1

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
    edge_index = [] # edge list source, destination

    for i in range(0,num_agents):
        for j in closest_neighbors_idx[i]:
            edge_index.append([i,j])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous() #transpose

def get_edge_attributes(edge_index, pos_list):
    edge_attributes = []

    for source_idx, dest_idx in zip(edge_index[0], edge_index[1]):
        source_pos = pos_list[source_idx]
        dest_pos = pos_list[dest_idx]
        edge_attributes.append(np.array([source_pos[0]-dest_pos[0], source_pos[1]-dest_pos[1]]))

    return edge_attributes

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
    # TODO have an otpion to choose method based on flags
    temp = bd_grid.clone()

    #centering
    center = bd_grid[:,1,k,k].clone() # 1 is to get bd, 0 for grid
    for i, cent in enumerate(center):
        bd_grid[i,1,:,:] -= cent
    bd_grid[:,1,:,:]*= (1-bd_grid[:,0,:,:]) # set obstacles as 0

    #normalize
    bd_grid[:, 1,:,:]/=k
    bd_grid[:,1,:,:] = torch.maximum(bd_grid[:,1,:,:], torch.tensor([-10]).to(device))
    bd_grid[:,1,:,:] = torch.minimum(bd_grid[:,1,:,:], torch.tensor([10]).to(device))
    return bd_grid

def create_data_object(pos_list, bd_list, grid, k, m, labels=np.array([])):
    num_agents = len(pos_list)
    x = [np.array([slice_maps(pos, grid, k), slice_maps(pos, bd, k)]) for (pos, bd) in zip(pos_list, bd_list)]

    m_closest_nborsIdx_list = [get_neighbors(m, pos, pos_list, k) for pos in pos_list] # (m,n)
    edge_index = convert_neighbors_to_edge_list(num_agents, m_closest_nborsIdx_list) # (2, num_edges)
    edge_attr = get_edge_attributes(edge_index, pos_list) # (num_edges,2 )

    # Tensorify
    x = torch.tensor(np.array(x), dtype=torch.float)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.int8) # up down left right stay (5 options)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = labels)

class MyOwnDataset(Dataset):
    def __init__(self, root, device, transform=None, pre_transform=None, pre_filter=None):
        self.k = 4 # padding size
        self.m = 5 # number of agents considered close
        self.size = float('inf')
        self.max_agents = 500
        # self.data_dictionaries = []
        self.length = 0
        self.device = device
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load_metadata()

    def load_metadata(self):
        if osp.isfile(osp.join(self.processed_dir, f"meta_data.pt")):
            self.length = torch.load(osp.join(self.processed_dir, f"meta_data.pt"))[0]

    @property
    def raw_file_names(self):
        #TODO: Change this based on npz name structure
        return [f"iterdata.npz"]

    @property
    def processed_file_names(self):
        file_names = []
        for i in range(self.length):
            file_names.append(f"data_{i}.pt")
        return  file_names # ["data_train.npz", "data_val.npz"]

    def download():
        raise ImportError


    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            print(raw_path)
            if osp.isfile(osp.join(self.processed_dir, f"data_{idx}.pt")):
                return
            # Read data from `raw_path`.
            cur_dataset = data_manipulator.PipelineDataset(raw_path, self.k, self.size, self.max_agents)
            for time_instance in tqdm(cur_dataset):
                # Graphify
                if not time_instance: break #idk why but the last one is None
                pos_list, labels, bd_list, grid = time_instance # (1), (2,n), (md,md): md=map dim with pad, n=num_agents

                curdata = create_data_object(pos_list, bd_list, grid, self.k, self.m, labels)
                curdata = apply_masks(len(curdata.x), curdata)
                torch.save(curdata, osp.join(self.processed_dir, f"data_{idx}.pt"))
                idx+=1

            self.length = idx
            meta = torch.tensor([self.length])
            torch.save(meta, osp.join(self.processed_dir, f"meta_data.pt"))

    def len(self):
        return self.length

    def get(self, idx):
        curdata = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        curdata = curdata.to(self.device)

        # normalize bd, normalize edge attributes
        edge_weights, bd_and_grids = curdata.edge_attr, curdata.x
        curdata.edge_attr = apply_edge_normalization(edge_weights)
        curdata.x = apply_bd_normalization(bd_and_grids, self.k, self.device)


        return curdata

# john = MyOwnDataset('map_data_big2d_new')
# john.get(5)
