import os.path as osp

import math
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
import pdb
import data_manipulator


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.k = 4 # padding size
        self.m = 5 # number of agents considered close
        self.size = float('inf')
        self.max_agents = 500
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["warehouse_10_20_10_2_2_train.npz", "warehouse_10_20_10_2_2_val.npz"]

    @property
    def processed_file_names(self):
        return ["data_1.pt", "data_2.pt"]

    def download():
        pass

    def slice_bd(self, pos, bd):
        """
        Turns a full size bd into a 2k+1, 2k+1
        param: pos, bd (current position and bd)
        output: sliced bd
        """
        r,c = pos[0], pos[1]
        k = self.k
        return bd[r-k:r+k+1,c-k:c+k+1] #TODO: adjust +1 or -1

    def get_neighbors(self, m, pos, pos_list):
        """
        Gets the m closest neighbors *within* 2k+1 bounding
        param: m, pos, pos_list
        output: list of indices of 5 closest
        """
        distances = cdist([pos], pos_list, 'euclidean')
        closest_5_in_box = []
        zipped_lists = zip(distances, pos_list, list(range(0,len(pos_list)))) #knit distances, pos, and idxs
        sorted_lists = sorted(zipped_lists, key = lambda t: t[0])
        k = self.k
        count, i = 0,0
        while count<m and i<len(pos_list):
            xdif = pos_list[i][0]-pos[0]
            ydif = pos_list[i][1]-pos[0]
            if xdif<k+1 and -k<xdif and ydif<k+1 and -k<ydif:
                count+=1
                closest_5_in_box.append(sorted_lists[i])
                # add to list if within bounding box
            i+=1
        return closest_5_in_box


    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            cur_dataset = data_manipulator.PipelineDataset(raw_path, self.k, self.size, self.max_agents)
            # cur_dataset is an array of all info for one file, cur_dataset[0] is first sample
            data_list = []
            for time_instance in cur_dataset:
                num_agents, pos_list, bd_list = time_instance # (1), (2,n), (md,md): md=map dim with pad, n=num_agents
                x = [(self.slice_bd(pos, bd)) for (pos,bd) in zip(pos_list, bd_list)] # (2k+1,2k+1,n)
                m_closest_nborsIdx_list = [(self.get_neighbors(self.m, pos, pos_list)) for pos in zip(pos_list)] # (5,n)



            pdb.set_trace()
            # n node graph
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # idx 0 = train, idx 1 = val
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data

john = MyOwnDataset('map_data')
