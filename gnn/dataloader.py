import os.path as osp

import math
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
import pdb
import pandas as pd # for loading status df

from torch_geometric.data import Data
from tqdm import tqdm
import os
import sys
from multiprocessing import Pool
import cProfile
import pstats
# from multiprocessing.dummy import Pool

# print(os.path.abspath(os.getcwd()))
# sys.path.insert(0, './data_collection/')
# import data_manipulator
from data_collection import data_manipulator
from custom_utils.custom_timer import CustomTimer

# def slice_maps(pos, curmap, k):
#     """
#     Turns a full size bd into a 2k+1, 2k+1
#     param: pos, bd (current position and bd)
#     output: sliced bd
#     """
#     r,c = pos[0], pos[1]
#     return curmap[r-k:r+k+1,c-k:c+k+1] 

# def get_neighbors(m, pos, pos_list, k):
#     """
#     Gets the m closest neighbors *within* 2k+1 bounding
#     param: m, pos, pos_list
#     output: list of indices of 5 closest
#     """
#     closest_5_in_box = []

#     distances = cdist([pos], pos_list, 'cityblock')[0]
#     zipped_lists = zip(distances, pos_list, list(range(0,len(pos_list)))) #knit distances, pos, and idxs
#     sorted_lists = sorted(zipped_lists, key = lambda t: t[0])
#     count, i = 0,0
#     while count<m and i<len(pos_list):
#         xdif = pos_list[i][0]-pos[0]
#         ydif = pos_list[i][1]-pos[1]
#         if abs(xdif) <= k and abs(ydif)<=k and (xdif!=0 and ydif!=0):
#             count+=1
#             closest_5_in_box.append(sorted_lists[i][2])
#             # add to list if within bounding box
#         i+=1
#     return closest_5_in_box

# def convert_neighbors_to_edge_list(num_agents, closest_neighbors_idx):
#     edge_index = [] # edge list source, destination

#     for i in range(0,num_agents):
#         for j in closest_neighbors_idx[i]:
#             edge_index.append([i,j])

#     return torch.tensor(edge_index, dtype=torch.long).t().contiguous() #transpose

# def get_edge_attributes(edge_index, pos_list):
#     edge_attributes = []

#     for source_idx, dest_idx in zip(edge_index[0], edge_index[1]):
#         source_pos = pos_list[source_idx]
#         dest_pos = pos_list[dest_idx]
#         edge_attributes.append(np.array([source_pos[0]-dest_pos[0], source_pos[1]-dest_pos[1]]))

#     return edge_attributes


# def get_idx_name(raw_path):
#     if 'val' in raw_path:
#         return 'val'
#     elif 'train' in raw_path:
#         return 'train'
#     else:
#         return 'unknownNPZ'

def apply_masks(data_len, curdata):
    tr_mask, te_mask = np.zeros(data_len), np.zeros(data_len)
    # tr_mask[:int((3/4)*data_len)] = 1
    # te_mask[int((3/4)*data_len):] = 1
    indices = torch.randperm(data_len) # (N)
    tr_mask[indices[:int((3/4)*data_len)]] = 1
    te_mask[indices[int((3/4)*data_len):]] = 1
    curdata.train_mask = torch.tensor(tr_mask, dtype=torch.bool)
    curdata.test_mask = torch.tensor(te_mask, dtype=torch.bool)
    return curdata

# def apply_edge_normalization(edge_weights):
#     """ edge_weights: (num_edges,2), the deltas in each direction
#     Note: These deltas can be negative, so exp(-edge_weights) does NOT work
#     """
#     # TODO have an option to choose method based on flags
#     edge_weights = torch.exp(-edge_weights)
#     return edge_weights

# def apply_bd_normalization(bd_grid, k, device):
#     center = bd_grid[:, 1, k, k].unsqueeze(1).unsqueeze(2)
#     bd_grid[:, 1, :, :] -= center
#     bd_grid[:, 1, :, :] *= (1 - bd_grid[:, 0, :, :])
#     bd_grid[:, 1, :, :] /= k
#     bd_grid[:, 1, :, :] = torch.clamp(bd_grid[:, 1, :, :], min=-10.0, max=10.0)

#     return bd_grid

def normalize_graph_data(data, k, edge_normalize="k", bd_normalize="center"):
    """Modifies data in place"""
    ### Normalize edge attributes
    # data.edge_attr (num_edges,2) the deltas in each direction which can be negative
    assert(edge_normalize in ["k"])
    if edge_normalize == "k":
        data.edge_attr /= k # Normalize edge attributes
    else:
        raise KeyError("Invalid edge normalization method: {}".format(edge_normalize))

    ### Normalize bd
    # pdb.set_trace()
    assert(bd_normalize in ["center"])
    bd_grid = data.x # (N,2,D,D)
    center = bd_grid[:, 1, k, k].unsqueeze(1).unsqueeze(2) # (N,1,1)
    bd_grid[:, 1, :, :] -= center
    bd_grid[:, 1, :, :] *= (1 - bd_grid[:, 0, :, :])
    bd_grid[:, 1, :, :] /= (2*k)
    bd_grid[:, 1, :, :] = torch.clamp(bd_grid[:, 1, :, :], min=-1.0, max=1.0)

    data.x = bd_grid
    return data


def create_data_object(pos_list, bd_list, grid, k, m, labels=np.array([])):
    """
    poslist: (N,2) positions
    bd_list: (N,W,H) bd's
    grid: (W,H) grid
    k: (int) local region size
    m: (int) number of closest neighbors to consider
    """
    num_agents = len(pos_list)
    # pdb.set_trace()
    # x = [np.array([slice_maps(pos, grid, k), slice_maps(pos, bd, k)]) for (pos, bd) in zip(pos_list, bd_list)]
    # x = np.array(x) # (N,2,W,H)
    ### Numpy advanced indexing to get all agent slices at once
    rowLocs = pos_list[:,0][:, None] # (N)->(N,1), Note doing (N)[:,None] adds an extra dimension
    colLocs = pos_list[:,1][:, None] # (N)->(N,1)

    x_mesh, y_mesh = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1), indexing='ij') # Each is (D,D)
    # Adjust indices to gather slices
    x_mesh = x_mesh[None, :, :] + rowLocs[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
    y_mesh = y_mesh[None, :, :] + colLocs[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
    grid_slices = grid[x_mesh, y_mesh] # (N,D,D)
    bd_slices = bd_list[np.arange(num_agents)[:,None,None], x_mesh, y_mesh] # (N,D,D)
    node_features = np.stack([grid_slices, bd_slices], axis=1) # (N,2,D,D)

    # pdb.set_trace()
    agent_indices = np.repeat(np.arange(num_agents)[None,:], axis=0, repeats=m).T # (N,N), each row is 0->num_agents
    deltas = pos_list[:, None, :] - pos_list[None, :, :] # (N,1,2) - (1,N,2) -> (N,N,2), the difference between each agent
    dists = np.linalg.norm(deltas, axis=2, ord=1) # (N,N), the distance between each agent
    fov_dist = np.any(np.abs(deltas) > k, axis=2) # (N,N,2)->(N,N) bool for if the agent is within the field of view
    dists[fov_dist] = np.inf # Set the distance to infinity if the agent is out of the field of view
    np.fill_diagonal(dists, np.inf) # Set the distance to itself to infinity
    closest_neighbors = np.argsort(dists, axis=1)[:, :m] # (N,m), the indices of the 4 closest agents
    distance_of_neighbors = dists[np.arange(num_agents)[:,None],closest_neighbors] # (N,m)
    
    # agent_inds = np.arange(num_agents)[:, None] # (N,1)
    neighbors_and_source_idx = np.stack([agent_indices, closest_neighbors]) # (2,N,m), 0 stores source agent, 1 stores neigbhor
    selection = distance_of_neighbors != np.inf # (N,m)
    edge_indices = neighbors_and_source_idx[:, selection] # (2, num_edges), [:,i] corresponds to (source, neighbor)
    edge_features = deltas[edge_indices[0], edge_indices[1]] # (num_edges,2), the difference between each agent

    # test_neighbors = []
    # totalCount = 0
    # for i in range(num_agents):
    #     cur_delta = np.abs(pos_list - pos_list[i]) # (N,2)
    #     dists = np.sum(cur_delta, axis=1) # (N)
    #     tmp = np.argsort(dists)[:m] # (m,)
    #     curN = []
    #     for j in tmp:
    #         if j == i or cur_delta[j][0] > k or cur_delta[j][1] > k:
    #             continue
    #         curN.append(j)
    #         totalCount+=1
    #     test_neighbors.append(curN)

    # m_closest_nborsIdx_list = [get_neighbors(m, pos, pos_list, k) for pos in pos_list] # (m,n) THE OUTPUT LOOKS WRONG
    # m_closest_nborsIdx_list = test_neighbors
    # edge_index = convert_neighbors_to_edge_list(num_agents, m_closest_nborsIdx_list) # (2, num_edges)
    # edge_attr = get_edge_attributes(edge_index, pos_list) # (num_edges,2 )
    # x = torch.tensor(np.array(x), dtype=torch.float)
    # edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.int8) # up down left right stay (5 options)
    # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = labels)

    return Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=torch.tensor(edge_indices, dtype=torch.long), 
                edge_attr=torch.tensor(edge_features, dtype=torch.float), 
                y = torch.tensor(labels, dtype=torch.int8))

class MyOwnDataset(Dataset):
    def __init__(self, root, exp_folder, iternum, 
                mapNpzFile, bdNpzFolder, pathNpzFolders,
                num_cores=20, transform=None, pre_transform=None, pre_filter=None, generate_initial=True):
        self.mapNpzFile = mapNpzFile
        self.bdNpzFolder = bdNpzFolder
        self.pathNpzFolders = pathNpzFolders
        self._raw_file_names = None # Use this to avoid recomputing raw_file_names everytime

        self.ct = CustomTimer()

        self.num_cores = num_cores
        self.k = 4 # padding size
        self.m = 5 # number of agents considered close
        self.size = float('inf')
        self.max_agents = 500
        self.exp_folder = exp_folder
        self.iternum = iternum
        self.current_iter_folder = f"{self.exp_folder}/iter{self.iternum}"
        # self.data_dictionaries = []
        self.length = 0
        # self.device = device
        self.generate_initial = generate_initial
        self.df = None
        super().__init__(root, transform, pre_transform, pre_filter) # this automatically calls process()
        # self.load_metadata()

    # def load_metadata(self):
    #     if osp.isfile(osp.join(self.processed_dir, f"meta_data.pt")):
    #         self.length = torch.load(osp.join(self.processed_dir, f"meta_data.pt"))[0]
    #     else:
    #         self.length = 0 
    #     return self.length

    @property
    def processed_dir(self) -> str:
        # return osp.join(self.exp_folder, 'processed')
        return osp.join(self.current_iter_folder, 'processed')

    def load_status_data(self):
        # self.df_path = osp.join(self.processed_dir, f"../status_data.csv")
        self.df_path = f"{self.processed_dir}/../status_data.csv"
        if osp.isfile(self.df_path):
            self.df = pd.read_csv(self.df_path)
        else:
            self.df = pd.DataFrame(columns=["npz_path", "pt_path", "status", "num_pts", "loading_time", "processing_time"])

    @property
    def raw_dir(self) -> str:
        raise NotImplementedError("Should not be called!")
    
    @property
    def raw_paths(self):
        return self.raw_file_names

    @property
    def num_classes(self) -> int:
        """REQUIRED: Otherwise dataset will iterate through everything really slowly"""
        return 5

    @property
    def raw_file_names(self):
        if self._raw_file_names is None:
            # npz_paths = os.listdir(self.exp_folder+"/labels/raw/")
            # npz_paths = os.listdir(f"{self.exp_folder}/iter{self.iternum}/eecbs_npzs/")
            # return npz_paths
            # pdb.set_trace()
            npz_paths = []
            for folder in self.pathNpzFolders:
            #     npz_paths.extend(os.listdir(folder))
                npz_paths.extend([os.path.join(folder, file) for file in os.listdir(folder)])
            self._raw_file_names = npz_paths
        # pdb.set_trace()
        print(len(self._raw_file_names))
        return self._raw_file_names

    @property
    def processed_file_names(self):
        file_names = []
        for i in range(self.length):
            file_names.append(f"data_{i}.pt")
        return file_names 

    @property
    def has_download(self) -> bool:
        """Need to define as parent Dataset checks this"""
        return False
    # def download():
    #     raise ImportError

    def create_and_save_graph(self, idx, time_instance):
        # Graphify
        if not time_instance: 
            return #idk why but the last one is None
        pos_list, labels, bd_list, grid = time_instance # (1), (2,n), (md,md): md=map dim with pad, n=num_agents

        curdata = create_data_object(pos_list, bd_list, grid, self.k, self.m, labels)
        curdata = apply_masks(len(curdata.x), curdata) # Adds train and test masks to data
        # torch.save(curdata, osp.join(self.processed_dir, f"data_{idx}.pt"))
        return curdata
    
    def process_single(self, npz_path):
        """
        Still in progress, do not use!
        """
        shared_lock = self.shared_lock
        assert(npz_path.endswith(".npz"))
        map_name = npz_path.split("/")[-1].removesuffix("_paths.npz")
        shared_lock.acquire()
        df_row = self.df.loc[self.df['npz_path'] == npz_path]
        assert(len(df_row) <= 1)
        if len(df_row) == 1:
            if df_row.iloc[0]["status"] == "processed":
                print(f"Skipping: {npz_path}")
            else:
                print(f"WARNING: Unexpected status for {map_name}: {df_row.iloc[0]['status']}")
        shared_lock.release()

        bdNpzFile = f"{self.bdNpzFolder}/{map_name}_bds.npz"
        cur_dataset = data_manipulator.PipelineDataset(self.mapNpzFile, bdNpzFile, npz_path, self.k, self.size, self.max_agents)
        print(f"Loading: {npz_path} of size {len(cur_dataset)}")

        tmp = []
        for t in tqdm(range(len(cur_dataset))):
            time_instance = cur_dataset[t]
            tmp.append(self.create_and_save_graph(idx+t, time_instance))
        torch.save(tmp, osp.join(self.processed_dir, f"data_{map_name}.pt"))
        idx += len(cur_dataset)

        new_df = pd.DataFrame.from_dict({"npz_path": [npz_path],
                                        "pt_path": [f"data_{map_name}.pt"],
                                        "status": ["processed"], 
                                        "num_pts": [len(cur_dataset)],
                                        "loading_time": [self.ct.getTimes("Loading", "list")[-1]], 
                                        "processing_time": [self.ct.getTimes("Processing", "list")[-1]]})
        
        shared_lock.acquire()
        if len(self.df) == 0:
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.df.to_csv(self.df_path, index=False)
        shared_lock.release()
        raise NotImplementedError("Not implemented yet")


    def process(self):
        self.load_status_data()
        if self.iternum==0 and not self.generate_initial:
            return #if pt's are made already skip the first iteration of pt making
        # idx = self.load_metadata()
        idx = 0
        print(f"Num cores: {self.num_cores}")
        for npz_path in self.raw_paths: #TODO check if new npzs are read
            # raw_path = "data_collection/data/logs/EXP_Test/iter0/eecbs_npzs/brc202d_paths.npz"
            # if "maps" in raw_path or "bds" in raw_path:
            #     continue
            if not npz_path.endswith(".npz"):
                continue
            # pdb.set_trace()
            map_name = npz_path.split("/")[-1].removesuffix("_paths.npz")
            # cur_path_iter = raw_path.split("_")[-1][:-4]
            # if not (int(cur_path_iter)==self.iternum):
            #     continue
            df_row = self.df.loc[self.df['npz_path'] == npz_path]
            assert(len(df_row) <= 1)
            if len(df_row) == 1:
                if df_row.iloc[0]["status"] == "processed":
                    print(f"Skipping: {npz_path}")
                    idx+=df_row.iloc[0]["num_pts"]
                    continue

            # cur_dataset = data_manipulator.PipelineDataset(raw_path, self.k, self.size, self.max_agents)
            bdNpzFile = f"{self.bdNpzFolder}/{map_name}_bds.npz"
            self.ct.start("Loading")
            cur_dataset = data_manipulator.PipelineDataset(self.mapNpzFile, bdNpzFile, npz_path, self.k, self.size, self.max_agents)
            print(f"Loading: {npz_path} of size {len(cur_dataset)}")
            # idx_list = np.array(range(len(cur_dataset)))+int(idx)

            self.ct.stop("Loading")
            self.ct.start("Processing")
            # pr = cProfile.Profile()
            # pr.enable()
            # tmp = []
            for t in tqdm(range(len(cur_dataset))):
                time_instance = cur_dataset[t]
                torch.save(self.create_and_save_graph(t, time_instance),
                            osp.join(self.processed_dir, f"data_{map_name}_{t}.pt"))
            # Save tmp to pt
            # torch.save(tmp, osp.join(self.processed_dir, f"data_{map_name}.pt"))
            idx += len(cur_dataset)
            self.ct.stop("Processing")
            # pr.disable()
            # ps = pstats.Stats(pr).sort_stats('cumtime')
            # ps.print_stats(10)
            # self.ct.start("Parallel Processing")
            ### Note: Multiprocessing seems slower than single processing
            # with Pool(self.num_cores) as p: #change number of workers later
            #     p.starmap(self.create_and_save_graph, zip(range(len(cur_dataset)), cur_dataset))
            # self.ct.stop("Parallel Processing")

            self.ct.printTimes()
            new_df = pd.DataFrame.from_dict({"npz_path": [npz_path],
                                            "pt_path": [f"data_{map_name}"],
                                            "status": ["processed"], 
                                            "num_pts": [len(cur_dataset)],
                                            "loading_time": [self.ct.getTimes("Loading", "list")[-1]], 
                                            "processing_time": [self.ct.getTimes("Processing", "list")[-1]]})
            if len(self.df) == 0:
                self.df = new_df
            else:
                self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df.to_csv(self.df_path, index=False)
        self.length = idx

        ### Get indices and files
        # self.order_of_indices looks like [0, 213, 457, 990, 1405, ...]
        # self.order_of_files looks like [map1, map2, map2, map4, map5, ...]
        # pdb.set_trace()
        self.order_of_indices = [0] # start with 0
        self.order_of_files = []
        # self.order_to_loaded_pt = []
        for row in self.df.iterrows():
            pt_path = row[1]["pt_path"]
            self.order_of_files.append(pt_path)
            self.order_of_indices.append(row[1]["num_pts"])
            # self.order_to_loaded_pt.append(torch.load(osp.join(self.processed_dir, pt_path)))
        self.order_of_indices.pop() # Remove the last element since it is the sum of all elements
        self.order_of_indices = np.cumsum(self.order_of_indices) # (num_maps)
        # pdb.set_trace()

        # for row in self.df.iterrows():
        #     self.order_to_loaded_pt.append(torch.load(osp.join(self.processed_dir, row[1]["pt_path"])))
        
    def len(self):
        """Require to override Dataset len()"""
        return self.length

    def get(self, idx):
        """Require to override Dataset get()"""
        # pdb.set_trace()
        # assert(self.df is not None and len(self.df) > 0)
        which_file_index = np.searchsorted(self.order_of_indices, idx, side='right')-1 # (num_maps)
        # data_file = self.order_of_files[which_file_index]
        data_idx = idx-self.order_of_indices[which_file_index]
        assert(data_idx >= 0)
        filename = f"{self.order_of_files[which_file_index]}_{data_idx}.pt"
        curdata = torch.load(osp.join(self.processed_dir, filename))
        # curdata = torch.load(osp.join(self.processed_dir, data_file))[data_idx]
        # curdata = self.order_to_loaded_pt[which_file_index][data_idx]
        # curdata = torch.load(osp.join(self.processed_dir, f"data_{data_file}.pt"))[data_idx]
        # curdata = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        # curdata = curdata.to(self.device) # Do not move it to device here, this slows stuff down and prevents pin_memory

        # normalize bd, normalize edge attributes
        # edge_weights, bd_and_grids = curdata.edge_attr, curdata.x
        # curdata.edge_attr = apply_edge_normalization(edge_weights)
        # curdata.x = apply_bd_normalization(bd_and_grids, self.k, self.device)

        return normalize_graph_data(curdata, self.k, edge_normalize="k", bd_normalize="center")

# john = MyOwnDataset('map_data_big2d_new')
# john.get(5)
