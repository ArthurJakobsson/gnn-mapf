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


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.k = 4 # padding size
        self.m = 5 # number of agents considered close
        self.size = float('inf')
        self.max_agents = 500
        self.length = 0
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["warehouse_10_20_10_2_2_train.npz", "warehouse_10_20_10_2_2_val.npz"]

    @property
    def processed_file_names(self):
        return ["data_train.pt", "data_val.pt"]

    def download():
        pass

    def slice_maps(self, pos, curmap):
        """
        Turns a full size bd into a 2k+1, 2k+1
        param: pos, bd (current position and bd)
        output: sliced bd
        """
        r,c = pos[0], pos[1]
        k = self.k
        return curmap[r-k:r+k+1,c-k:c+k+1] #TODO: adjust +1 or -1

    def get_neighbors(self, m, pos, pos_list):
        """
        Gets the m closest neighbors *within* 2k+1 bounding
        param: m, pos, pos_list
        output: list of indices of 5 closest
        """
        closest_5_in_box = []

        distances = cdist([pos], pos_list, 'euclidean')[0]
        zipped_lists = zip(distances, pos_list, list(range(0,len(pos_list)))) #knit distances, pos, and idxs
        sorted_lists = sorted(zipped_lists, key = lambda t: t[0])
        k = self.k
        count, i = 0,0
        while count<m and i<len(pos_list):
            xdif = pos_list[i][0]-pos[0]
            ydif = pos_list[i][1]-pos[1]
            if abs(xdif) <= k and abs(ydif): #TODO: check bounding box
                count+=1
                closest_5_in_box.append(sorted_lists[i][2])
                # add to list if within bounding box
            i+=1
        # pdb.set_trace()
        return closest_5_in_box

    def convert_neighbors_to_edge_list(self, num_agents, closest_neighbors_idx):
        edge_index = [] # edge list source, destination

        for i in range(0,num_agents):
            for j in closest_neighbors_idx[i]:
                edge_index.append([i,j])

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous() #transpose

    def get_edge_attributes(self, edge_index, pos_list):
        edge_attributes = []

        for source_idx, dest_idx in zip(edge_index[0], edge_index[1]):
            source_pos = pos_list[source_idx]
            dest_pos = pos_list[dest_idx]
            # pdb.set_trace()
            edge_attributes.append(np.array([source_pos[0]-dest_pos[0], source_pos[1]-dest_pos[1]]))

        return edge_attributes

    def get_idx_name(self, raw_path):
        if 'val' in raw_path:
            return 'val'
        elif 'train' in raw_path:
            return 'train'
        else:
            return 'unknownNPZ'

    def process(self):
        for raw_path in self.raw_paths:
            print(raw_path)
            idx = self.get_idx_name(raw_path)
            # if osp.isfile(osp.join(self.processed_dir, f"data_{idx}.pt")):
                # return
            # Read data from `raw_path`.
            cur_dataset = data_manipulator.PipelineDataset(raw_path, self.k, self.size, self.max_agents)
            # cur_dataset is an array of all info for one file, cur_dataset[0] is first sample
            data_list = []
            # pdb.set_trace()
            count = 0
            for time_instance in tqdm(cur_dataset):
                # Graphify
                count+=1
                if count==100: break
                if not time_instance: break #idk why but the last one is None
                pos_list, labels, bd_list, grid = time_instance # (1), (2,n), (md,md): md=map dim with pad, n=num_agents
                num_agents = len(pos_list)
                x = [(np.vstack([self.slice_maps(pos, grid),self.slice_maps(pos, bd)])) for (pos,bd) in zip(pos_list, bd_list)] # (2, 2k+1,2k+1,n) both grid and bds for each agent's window
                m_closest_nborsIdx_list = [self.get_neighbors(self.m, pos, pos_list) for pos in pos_list] # (m,n)
                edge_index = self.convert_neighbors_to_edge_list(num_agents, m_closest_nborsIdx_list) # (2, num_edges)
                edge_attr = self.get_edge_attributes(edge_index, pos_list) # (2, num_edges)

                # Tensorify
                x = torch.tensor(np.array(x), dtype=torch.float)
                edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.int8) # up down left right stay (5 options)
                curdata = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = labels)
                data_list.append(curdata)


            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue

            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)
            self.length += len(data_list)
            pdb.set_trace()
            data_list = torch.tensor(data_list)
            # each file is either train or val and contains many graphs
            print("Saving File ", osp.join(self.processed_dir, f"data_{idx}.pt"))
            torch.save(data_list, osp.join(self.processed_dir, f"data_{idx}.pt"))

    def len(self):
        data_train = torch.load(osp.join(self.processed_dir, f"data_train.pt"))
        data_val = torch.load(osp.join(self.processed_dir, f"data_val.pt"))
        self.length = len(data_train)+len(data_val)
        return self.length

    def get(self, idx):
        data_train = torch.load(osp.join(self.processed_dir, f"data_train.pt"))
        data_val = torch.load(osp.join(self.processed_dir, f"data_val.pt"))
        pdb.set_trace()
        data = torch.cat(data_train, data_val)
        return data[idx]

# john = MyOwnDataset('map_data')
