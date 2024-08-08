import torch
from torch_geometric.data import Data, Dataset, download_url
import numpy as np
import pdb
import pandas as pd # for loading status df

from tqdm import tqdm
import os
import os.path as osp
import argparse
import sys
from multiprocessing import Pool # This uses processes
# from multiprocessing.dummy import Pool # This uses threads instead of processes
import cProfile
import pstats

from data_collection import data_manipulator
from custom_utils.custom_timer import CustomTimer


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
    assert(data.x[:,1,k,k].all() == 0) # Make sure all agents are on empty space
    assert(data.x[:,1].max() <= 1.0 and data.x[:,1].min() >= -1.0) # Make sure all agents are on empty space
    assert(data.x[:,0,k,k].all() == 0) # Make sure all agents are on empty space
    return data

def create_data_object(pos_list, bd_list, grid, k, m, goal_locs, labels=np.array([])):
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
    assert(grid[pos_list[:,0], pos_list[:,1]].all() == 0) # Make sure all agents are on empty space

    x_mesh, y_mesh = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1), indexing='ij') # Each is (D,D)
    # Adjust indices to gather slices
    x_mesh = x_mesh[None, :, :] + rowLocs[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
    y_mesh = y_mesh[None, :, :] + colLocs[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
    grid_slices = grid[x_mesh, y_mesh] # (N,D,D)
    bd_slices = bd_list[np.arange(num_agents)[:,None,None], x_mesh, y_mesh] # (N,D,D)
    # pdb.set_trace()
    agent_pos = np.zeros((grid.shape[0], grid.shape[1])) # (W,H)
    agent_pos[rowLocs, colLocs] = 1 # (W,H)
    agent_pos_slices = agent_pos[x_mesh, y_mesh] # (N,D,D)

    # TODO get the best location to go next, just according to the bd
    # NOTE: because we pad all bds with a large number, 
    # we should be able to get the up, down, left and right of each bd without fear of invalid indexing
    best_moves = np.zeros((num_agents, 5)) # (N, [up, left, down, right, stop])
    bd_tmp = np.copy(bd_slices)
    x_mesh2, y_mesh2 = np.meshgrid(np.arange(-1,1+1), np.arange(-1,1+1), indexing='ij') # assumes k at least 1; getting a 3x3 grid centered at the same place
    bd_tmp = bd_tmp[x_mesh2, y_mesh2]
    # set diagonal entries to a big number
    bd_tmp[:,0,0] = 1073741823
    bd_tmp[:,0,2] = 1073741823
    bd_tmp[:,2,0] = 1073741823
    bd_tmp[:,2,2] = 1073741823
    mins = np.min(bd_tmp, axis=(1,2)) # take a min
    flattened = np.reshape(bd_tmp, (-1, 9))
    flattened = flattened[:,[(1,3,7,5,4)]]

    # Create a boolean array where each element is True if it is the minimum in its row
    min_indices = flattened == mins[:, None]
    min_indices = np.array([min_indices[i][i] for i in range(len(min_indices))]) # (N, 5) non-unique argmin solution
    # min_indices = bd_tmp == mins[:, None]
    # min_indices = np.pad(min_indices, ((0,0),(k-1,k-1),(k-1,k-1)), constant_values=True) # (N,D,D) no-unique argmin solution
    
    # goalRowLocs, goalColLocs= goal_locs[:,0][:, None], goal_locs[:,1][:, None]  # (N,1), (N,1)
    # goal_pos = np.zeros((grid.shape[0], grid.shape[1])) # (W,H)
    # NOTE: for 1 hot goal version
    # goal_pos = np.zeros((num_agents, grid.shape[0], grid.shape[1])) # (N,W,H)
    # goal_pos[np.arange(num_agents), goalRowLocs, goalColLocs] = 1 # (N,W,H)
    # for i in range(num_agents):
    #     goal_pos[i, goalRowLocs[i], goalColLocs[i]] = 1
    
    # NOTE: for all agents on their goal turn to 1 version
    # matches = (rowLocs == goalRowLocs) & (colLocs == goalColLocs) # Compare row and column locations
    
    # goal_pos[rowLocs[matches], colLocs[matches]] = 1 # Set goal positions to 1 where matches are found
    # goal_pos_slices = goal_pos[x_mesh, y_mesh] # (N,D,D)
    # pdb.set_trace()
    # assert(goal_pos_slices.shape==bd_slices.shape)
    # pdb.set_trace()
    # if pos_locs == goal_loc add a 1 otherwise keep it at a 0
    # node_features = np.stack([grid_slices, bd_slices, goal_pos_slices], axis=1) # (N,3,D,D)
    node_features = np.stack([grid_slices, bd_slices, agent_pos_slices], axis=1) # (N,3,D,D)

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
    # pdb.set_trace()
    assert(node_features[:,0,k,k].all() == 0) # Make sure all agents are on empty space

    return Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=torch.tensor(edge_indices, dtype=torch.long), 
                edge_attr=torch.tensor(edge_features, dtype=torch.float), 
                y = torch.tensor(labels, dtype=torch.int8),bd_suggestion=best_moves)


class MyOwnDataset(Dataset):
    def __init__(self, mapNpzFile, bdNpzFolder, pathNpzFolder,
                processedOutputFolder, num_cores, k, m):
        if num_cores != 1:
            raise NotImplementedError("Multiprocessing not supported yet")
        
        self.mapNpzFile = mapNpzFile
        self.bdNpzFolder = bdNpzFolder
        self.pathNpzFolder = pathNpzFolder
        self.processedOutputFolder = processedOutputFolder
        if not osp.exists(self.processedOutputFolder):
            os.makedirs(self.processedOutputFolder)
        self._raw_file_names = None # Use this to avoid recomputing raw_file_names everytime

        self.ct = CustomTimer()

        self.num_cores = num_cores
        self.k = k # padding size
        self.m = m # number of agents considered close
        self.size = float('inf')
        self.max_agents = 500

        # self.data_dictionaries = []
        self.length = 0
        self.df = None
        super().__init__() # this automatically calls process()
        self.custom_process()

    @property
    def processed_dir(self) -> str:
        # return osp.join(self.exp_folder, 'processed')
        # return osp.join(self.current_iter_folder, 'processed')
        return self.processedOutputFolder

    def load_status_data(self):
        folder_name = osp.basename(osp.normpath(self.processedOutputFolder)) # Gets the name of processed folder
        self.df_path = f"{self.processed_dir}/../status_data_{folder_name}.csv"
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
            # npz_paths = []
            # for folder in self.pathNpzFolders:
            # #     npz_paths.extend(os.listdir(folder))
            #     npz_paths.extend([os.path.join(folder, file) for file in os.listdir(folder)])
            # self._raw_file_names = npz_paths
            self._raw_file_names = [os.path.join(self.pathNpzFolder, file) for file in os.listdir(self.pathNpzFolder)]
        # print("Num npz paths to process: ", len(self._raw_file_names))
        return self._raw_file_names

    # @property
    # def processed_file_names(self):
    #     file_names = []
    #     for i in range(self.length):
    #         file_names.append(f"data_{i}.pt")
    #     return file_names 

    @property
    def has_process(self) -> bool:
        """Need to define as parent Dataset checks this"""
        return False

    @property
    def has_download(self) -> bool:
        """Need to define as parent Dataset checks this"""
        return False
    # def download():
    #     raise ImportError

    def create_and_save_graph(self, idx, time_instance):
        # Graphify
        # if not time_instance: 
        #     return #idk why but the last one is None
        assert(time_instance is not None)
        pos_list, labels, bd_list, grid, goal_locs = time_instance

        curdata = create_data_object(pos_list, bd_list, grid, self.k, self.m, goal_locs, labels)
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


    def custom_process(self):
        self.load_status_data() # This loads self.df which is used for checking if should process or skip

        if self.pathNpzFolder is not None: # Run process
            # idx = 0
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
                        # idx+=df_row.iloc[0]["num_pts"]
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
                # idx += len(cur_dataset)
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
            # self.length = idx
        
        self.length = self.df["num_pts"].sum()
        print("Loading dataset with length: ", self.length)

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


### Example run
"""
python -m gnn.dataloader --mapNpzFile=data_collection/data/benchmark_data/constant_npzs/all_maps.npz \
      --bdNpzFolder=data_collection/data/benchmark_data/constant_npzs \
      --pathNpzFolder=data_collection/data/logs/EXP_Test3/iter0/eecbs_npzs \
      --processedFolder=data_collection/data/logs/EXP_Test3/iter0/processed \
"""
if __name__ == "__main__":
    """
    This file takes in npzs and processes them into pt files.

    We assume the following file structure:
    Inputs:
    - mapNpzFile: e.g. all_maps.npz with all_maps[MAPNAME] = grid_map (H,W)
    - bdNpzFolder: Folder containing [MAPNAME]_bds.npz    
    - pathNpzFolder: Folder containing [MAPNAME]_paths.npz
    Outputs:
    - processedFolder: Folder to save processed data, will contain [MAPNAME]_[idx].pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapNpzFile", help="map npz file", type=str, required=True)
    parser.add_argument("--bdNpzFolder", help="bd npz folder", type=str, required=True)
    parser.add_argument("--pathNpzFolder", help="path npz folder", type=str, required=True)
    parser.add_argument("--processedFolder", help="processed folder to save pt", type=str, required=True)
    parser.add_argument("--k", help="window size", type=int)
    parser.add_argument("--m", help="num_nearby_agents", type=int)
    args = parser.parse_args()

    dataset = MyOwnDataset(mapNpzFile=args.mapNpzFile, bdNpzFolder=args.bdNpzFolder, 
                        pathNpzFolder=args.pathNpzFolder, processedOutputFolder=args.processedFolder,
                        num_cores=1, k=args.k, m=args.m)
