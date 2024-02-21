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
        self.data_dictionaries = []
        self.length = 0
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load_dictionaries()



    def load_dictionaries(self):
        # if files exist cache them
        for raw_path in self.raw_paths:
            idx = self.get_idx_name(raw_path)
            file_path = osp.join(self.processed_dir, f"data_{idx}.npz")
            if osp.isfile(osp.join(self.processed_dir, f"data_{idx}.npz")):
                self.data_dictionaries.append(np.load(file_path, allow_pickle=True))

        # Sum length of all timesteps
        for datum in self.data_dictionaries:
            self.length += len(datum['x'])


    @property
    def raw_file_names(self):
        return ["warehouse_10_20_10_2_2_train.npz", "warehouse_10_20_10_2_2_val.npz"]

    @property
    def processed_file_names(self):
        return ["data_train.npz", "data_val.npz"]

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
            if osp.isfile(osp.join(self.processed_dir, f"data_{idx}.pt")):
                return
            # Read data from `raw_path`.
            cur_dataset = data_manipulator.PipelineDataset(raw_path, self.k, self.size, self.max_agents)
            # cur_dataset is an array of all info for one file, cur_dataset[0] is first sample
            data_dict = {'x':[], 'edge_index': [], 'edge_attr':[], 'labels': [], 'length': 0}
            # pdb.set_trace()
            count = 0
            for time_instance in tqdm(cur_dataset):
                # Graphify
                if not time_instance: break #idk why but the last one is None
                pos_list, labels, bd_list, grid = time_instance # (1), (2,n), (md,md): md=map dim with pad, n=num_agents
                num_agents = len(pos_list)
                x = [(np.vstack([self.slice_maps(pos, grid),self.slice_maps(pos, bd)])).flatten() for (pos,bd) in zip(pos_list, bd_list)] # (2, 2k+1,2k+1,n) both grid and bds for each agent's window
                m_closest_nborsIdx_list = [self.get_neighbors(self.m, pos, pos_list) for pos in pos_list] # (m,n)
                edge_index = self.convert_neighbors_to_edge_list(num_agents, m_closest_nborsIdx_list) # (2, num_edges)
                edge_attr = self.get_edge_attributes(edge_index, pos_list) # (num_edges,2 )


                # Tensorify
                x = torch.tensor(np.array(x), dtype=torch.float)
                edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.int8) # up down left right stay (5 options)
                # pdb.set_trace()
                # Add to dict
                data_dict['x'].append(x)
                data_dict['edge_index'].append(edge_index)
                data_dict['edge_attr'].append(edge_attr)
                data_dict['labels'].append(labels)

                # curdata = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = labels)
                # data_list.append(curdata)


            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue

            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)
            self.length += len(data_dict['x'])
            data_dict['length'] = len(data_dict['x'])
            # pdb.set_trace()
            # pdb.set_trace()
            # each file is either train or val and contains many graphs
            print("Saving File ", osp.join(self.processed_dir, f"data_{idx}.pt"))
            np.savez(osp.join(self.processed_dir, f"data_{idx}.npz"), **data_dict)

    def len(self):
        return self.length

    def get(self, idx):
        timestep_count = 0
        curdata = None
        for file_dict in self.data_dictionaries:
            timestep_count+=file_dict['length']
            if timestep_count < idx:
                pass
            else:
                relative_idx = timestep_count-idx-1
                x = file_dict['x'][relative_idx]
                edge_index = file_dict['edge_index'][relative_idx]
                edge_attr = file_dict['edge_attr'][relative_idx]
                labels = file_dict['labels'][relative_idx]
                # pdb.set_trace()
                curdata = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = labels)
                data_len = len(x)
                tr_mask, te_mask = np.zeros(data_len), np.zeros(data_len)
                tr_mask[:int((3/4)*data_len)] = 1
                te_mask[int((3/4)*data_len):] = 1
                curdata.train_mask = torch.tensor(tr_mask, dtype=torch.bool)
                curdata.test_mask = torch.tensor(te_mask, dtype=torch.bool)
                break

        return curdata

# john = MyOwnDataset('map_data')
