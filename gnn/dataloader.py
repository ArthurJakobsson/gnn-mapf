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
    r, c = pos[0], pos[1]
    return curmap[r-k:r+k+1, c-k:c+k+1]

def get_neighbors(m, pos, pos_list, k):
    distances = cdist([pos], pos_list, 'euclidean')[0]
    zipped_lists = zip(distances, pos_list, list(range(len(pos_list))))
    sorted_lists = sorted(zipped_lists, key=lambda t: t[0])

    closest_5_in_box = []
    count, i = 0, 0
    while count < m and i < len(pos_list):
        xdif, ydif = pos_list[i][0] - pos[0], pos_list[i][1] - pos[1]
        if abs(xdif) <= k and abs(ydif) <= k:
            count += 1
            closest_5_in_box.append(sorted_lists[i][2])
        i += 1
    return closest_5_in_box

def convert_neighbors_to_edge_list(num_agents, closest_neighbors_idx):
    edge_index = []
    for i in range(num_agents):
        for j in closest_neighbors_idx[i]:
            edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def get_edge_attributes(edge_index, pos_list):
    edge_attributes = np.array([pos_list[source_idx] - pos_list[dest_idx] for source_idx, dest_idx in zip(edge_index[0], edge_index[1])])
    return edge_attributes

def get_idx_name(raw_path):
    if 'val' in raw_path:
        return 'val'
    elif 'train' in raw_path:
        return 'train'
    else:
        return 'unknownNPZ'

def apply_masks(data_len, curdata):
    tr_mask = np.zeros(data_len, dtype=bool)
    te_mask = np.zeros(data_len, dtype=bool)
    tr_mask[:int((3/4) * data_len)] = True
    te_mask[int((3/4) * data_len):] = True
    curdata.train_mask = torch.tensor(tr_mask)
    curdata.test_mask = torch.tensor(te_mask)
    return curdata

def apply_edge_normalization(edge_weights):
    return torch.exp(-edge_weights)

def apply_bd_normalization(bd_grid, k, device):
    temp = bd_grid.clone()

    center = bd_grid[:, 1, k, k].unsqueeze(1).unsqueeze(2)
    bd_grid[:, 1, :, :] -= center
    bd_grid[:, 1, :, :] *= (1 - bd_grid[:, 0, :, :])
    bd_grid[:, 1, :, :] /= k
    bd_grid[:, 1, :, :] = torch.clamp(bd_grid[:, 1, :, :], min=-10.0, max=10.0)

    return bd_grid

def create_data_object(pos_list, bd_list, grid, k, m, labels=np.array([])):
    num_agents = len(pos_list)
    x = np.array([[slice_maps(pos, grid, k), slice_maps(pos, bd, k)] for pos, bd in zip(pos_list, bd_list)])
    m_closest_nborsIdx_list = [get_neighbors(m, pos, pos_list, k) for pos in pos_list]
    edge_index = convert_neighbors_to_edge_list(num_agents, m_closest_nborsIdx_list)
    edge_attr = torch.tensor(get_edge_attributes(edge_index, pos_list), dtype=torch.float)

    x = torch.tensor(x, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.int8)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)

class MyOwnDataset(Dataset):
    def __init__(self, root, device, transform=None, pre_transform=None, pre_filter=None):
        self.k = 4
        self.m = 5
        self.size = float('inf')
        self.max_agents = 500
        self.length = 0
        self.device = device
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load_metadata()

    def load_metadata(self):
        if osp.isfile(osp.join(self.processed_dir, "meta_data.pt")):
            self.length = torch.load(osp.join(self.processed_dir, "meta_data.pt"))[0]

    @property
    def raw_file_names(self):
        return ["warehouse_10_20_10_2_2_train.npz", "warehouse_10_20_10_2_2_val.npz"]

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(self.length)]

    def download(self):
        pass
    
    @property
    def num_classes(self) -> int:
        return 5

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            print(raw_path)
            if osp.isfile(osp.join(self.processed_dir, f"data_{idx}.pt")):
                return
            cur_dataset = data_manipulator.PipelineDataset(raw_path, self.k, self.size, self.max_agents)
            for time_instance in tqdm(cur_dataset):
                if not time_instance:
                    break
                pos_list, labels, bd_list, grid = time_instance
                curdata = create_data_object(pos_list, bd_list, grid, self.k, self.m, labels)
                curdata = apply_masks(len(curdata.x), curdata)
                torch.save(curdata, osp.join(self.processed_dir, f"data_{idx}.pt"))
                idx += 1

            self.length = idx
            meta = torch.tensor([self.length])
            torch.save(meta, osp.join(self.processed_dir, "meta_data.pt"))

    def len(self):
        return self.length

    def get(self, idx):
        curdata = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt")).to(self.device)
        curdata.edge_attr = apply_edge_normalization(curdata.edge_attr)
        curdata.x = apply_bd_normalization(curdata.x, self.k, self.device)
        return curdata

# john = MyOwnDataset('map_data_big2d_new')
# john.get(5)
