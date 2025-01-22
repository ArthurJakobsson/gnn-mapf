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
    curdata.train_mask = torch.from_numpy(tr_mask)
    curdata.test_mask = torch.from_numpy(te_mask)
    return curdata

def normalize_graph_data(data, k, edge_normalize="k", bd_normalize="center"):
    # pdb.set_trace()
    """Modifies data in place"""
    ### Normalize edge attributes
    # data.edge_attr (num_edges,2) the deltas in each direction which can be negative

    # assert(edge_normalize in ["k"])
    # if edge_normalize == "k":
    #     data.edge_attr = data.edge_attr.type(torch.FloatTensor) # convert dtype to float32
    #     data.edge_attr /= k # Normalize edge attributes
    # else:
    #     raise KeyError("Invalid edge normalization method: {}".format(edge_normalize))

    ### Normalize bd
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

def get_bd_prefs(pos_list, bds, range_num_agents):
    """
    pos_list: (N,2) positions
    bds: (N,W,H) bd's
    range_num_agents: (N) range of number of agents
    """
    x_mesh2, y_mesh2 = np.meshgrid(np.arange(-1,1+1), np.arange(-1,1+1), indexing='ij') # assumes k at least 1; getting a 3x3 grid centered at the same place
    # pdb.set_trace()
    x_mesh2 = x_mesh2[None, :, :] + np.expand_dims(pos_list[:,0], axis=(1,2)) #  -> (N,3,3)
    y_mesh2 = y_mesh2[None, :, :] + np.expand_dims(pos_list[:,1], axis=(1,2)) # -> (N,3,3)
    bd_subset = bds[range_num_agents[:,None,None], x_mesh2, y_mesh2] # (N,3,3)
    flattened = np.reshape(bd_subset, (-1, 9)) # (N,9) order (top to bot) left mid right, left mid right, left mid right
    flattened = flattened[:,(4,5,7,1,3)] # (N,5) consistent with NN
    flattened = flattened.astype(float) + np.random.random(flattened.shape)*1e-6 # Add noise to break ties
    # NOTE: Random noise is extremely import for PIBT to work it seems
    prefs = np.argsort(flattened, axis=1, kind="quicksort") # (N,5) Stop, Right, Down, Up, Left
    # pdb.set_trace()
    return prefs


def calculate_node_features(bd_slices, grid_slices, rowLocs, colLocs, goalRowLocs, goalColLocs, matches, x_mesh, y_mesh, 
                      grid, goal_locs, extra_layers, num_layers, num_agents):
    N,D = bd_slices.shape[0], bd_slices.shape[1]

    node_features = np.empty((N,num_layers,D,D),dtype=np.float32) # default num_layers = 2 (grid slices and bd slices)
    node_feature_idx = 0
    node_features[:,node_feature_idx] = grid_slices
    node_feature_idx +=1
    node_features[:,node_feature_idx] = bd_slices
    node_feature_idx +=1

    if extra_layers is not None:
        if "agent_locations" in extra_layers:
            agent_pos = np.zeros((grid.shape[0], grid.shape[1])) # (W,H)
            agent_pos[rowLocs, colLocs] = 1 # (W,H)
            agent_pos_slices = agent_pos[x_mesh, y_mesh] # (N,D,D)
            node_features[:,node_feature_idx] = agent_pos_slices
            node_feature_idx +=1
    
        if "agent_goal" in extra_layers:
            # NOTE: for 1 hot goal version
            goal_pos = np.zeros((num_agents, grid.shape[0], grid.shape[1])) # (N,W,H)   
            goal_pos[np.arange(num_agents), goalRowLocs, goalColLocs] = 1 # (N,W,H)
            node_features[:,node_feature_idx] = goal_pos
            node_feature_idx +=1
            
        if "near_goal_info" in extra_layers:
            rowLocs_g = goal_locs[:,0][:, None] # (N)->(N,1), Note doing (N)[:,None] adds an extra dimension
            colLocs_g = goal_locs[:,1][:, None] # (N)->(N,1)
            x_mesh_g, y_mesh_g = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1), indexing='ij') # Each is (D,D)
            # Adjust indices to gather slices
            x_mesh_g = x_mesh_g[None, :, :] + rowLocs_g[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
            y_mesh_g = y_mesh_g[None, :, :] + colLocs_g[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
            near_goal_grid_slices = grid[x_mesh_g, y_mesh_g]
            node_features[:,node_feature_idx] = near_goal_grid_slices
            node_feature_idx +=1
            
            not_at_goal_binary = np.zeros((grid.shape[0], grid.shape[1])) # (W,H)
            not_at_goal_binary[rowLocs[np.invert(matches)], colLocs[np.invert(matches)]] = 1 
            not_at_goal_slices = not_at_goal_binary[x_mesh_g, y_mesh_g] # (N,D,D)
            node_features[:,node_feature_idx] = not_at_goal_slices
            node_feature_idx +=1
        
        if "at_goal_grid" in extra_layers:
            goal_pos_binary = np.zeros((grid.shape[0], grid.shape[1])) # (W,H)
            # NOTE: for all agents on their goal turn to 1 version
            goal_pos_binary[rowLocs[matches], colLocs[matches]] = 1 # Set goal positions to 1 where matches are found
            goal_pos_slices = goal_pos_binary[x_mesh, y_mesh] # (N,D,D)
            assert(goal_pos_slices.shape==bd_slices.shape)
            node_features[:,node_feature_idx] = goal_pos_slices
            node_feature_idx +=1

    return node_features


# (2,num_edges), (num_edges,num_layers)
def calculate_edge_features(pos_list, num_agents, range_num_agents, m, k, priorities, num_priority_copies):
    agent_indices = np.repeat(np.arange(num_agents)[None,:], axis=0, repeats=m).T # (N,m), each row is 0->num_agents
    deltas = pos_list[:, None, :] - pos_list[None, :, :] # (N,1,2) - (1,N,2) -> (N,N,2), the difference between each agent

    ## Calculate the distance between each agent, einsum is faster than other options
    dists = np.einsum('ijk,ijk->ij', deltas, deltas, optimize='optimal').astype(float) # (N,N), the L2^2 distance between each agent
    # dists2 = np.linalg.norm(deltas, axis=2, ord=2) # (N,N), the distance between each agent
    # dists3 = np.sum(np.abs(deltas)**2, axis=2) # (N,N), the distance between each agent
    # assert(np.allclose(dists1, dists3)) # Make sure the two distance calculations are the same
    # assert(np.allclose(dists1, np.sqrt(dists2))) # Make sure the two distance calculations are the same

    fov_dist = np.any(np.abs(deltas) > k, axis=2) # (N,N,2)->(N,N) bool for if the agent is within the field of view
    dists[fov_dist] = np.inf # Set the distance to infinity if the agent is out of the field of view
    closest_neighbors = np.argsort(dists, axis=1, kind="quicksort")[:, 1:m+1] # (N,m), the indices of the 4 closest agents, ignore self
    # arg_dists = np.argpartition(dists, m+1, axis=1)
    # closest_neighbors = arg_dists[:,1:m+1]
    distance_of_neighbors = dists[range_num_agents[:,None],closest_neighbors] # (N,m)
    
    # neighbors_and_source_idx = np.stack([agent_indices, closest_neighbors]) # (2,N,m), 0 stores source agent, 1 stores neigbhor
    # selection = distance_of_neighbors != np.inf # (N,m)
    # edge_indices = neighbors_and_source_idx[:, selection] # (2,num_edges), [:,i] corresponds to (source, neighbor)
    # edge_features = deltas[edge_indices[0], edge_indices[1]] # (num_edges,2), the difference between each agent
    # edge_features = edge_features.astype(np.float32)

    # pdb.set_trace()
    # Priorities
    priorities_repeat = np.repeat(priorities[None,:], axis=0, repeats=num_agents)
    relative_priorities = -(priorities_repeat.T - priorities_repeat) # (N, N), i relative to j (positive is higher priority, 0 is highest priority)
    
    # normalize relative_priority in [-(num_agents-1), num_agents-1] to a value in [0, 1]
    relative_priorities = (relative_priorities + (num_agents-1)) / (2*(num_agents-1))

    neighbors_and_source_idx = np.stack([agent_indices, closest_neighbors]) # (2,N,m), 0 stores source agent, 1 stores neigbhor
    selection = distance_of_neighbors != np.inf # (N,m)
    edge_indices = neighbors_and_source_idx[:, selection] # (2,num_edges), [:,i] corresponds to (source, neighbor)

    relative_priority = relative_priorities[edge_indices[0], edge_indices[1]][:, None] # (num_edges,1)
    edge_features = np.concatenate((deltas[edge_indices[0], edge_indices[1]], # (num_edges,2), the position difference between agents
                                    np.repeat(relative_priority, num_priority_copies,axis=1)), axis=1) # (num_edges,num_priority_copies)
    
    return edge_indices, edge_features


def calculate_bd_pred_arr(grid_slices, rowLocs, colLocs, num_layers, num_agents, bd_pred):
    bd_pred_arr = None
    linear_dimensions = (grid_slices.shape[1]-2)**2 * num_layers
    if bd_pred is not None:
        # TODO get the best location to go next, just according to the bd
        # NOTE: because we pad all bds with a large number, 
        # we should be able to get the up, down, left and right of each bd without fear of invalid indexing
        # (N, [Stop, Right, Down, Up, Left])
        x_mesh2, y_mesh2 = np.meshgrid(np.arange(-1,1+1), np.arange(-1,1+1), indexing='ij') # assumes k at least 1; getting a 3x3 grid centered at the same place
        x_mesh2 = x_mesh2[None, :, :] + rowLocs[:, None, :] # (1, 3, 3) + (N, 1, 1) -> (N,3,3)
        y_mesh2 = y_mesh2[None, :, :] + colLocs[:, None, :] # (1, 3, 3) + (N, 1, 1) -> (N,3,3)
        bd_list = bd_list[np.arange(num_agents)[:,None,None], x_mesh2, y_mesh2] # (N,3,3)
        # set diagonal entries to a big number
        flattened = np.reshape(bd_list, (-1, 9)) # (N,9) # (order (top to bot) left mid right, left mid right, left mid right)
        flattened = flattened[:,[(4,5,7,1,3)]].reshape((-1,5)) # (N,5)

        # Create a boolean array where each element is True if it is the minimum in its row
        min_indices = flattened == flattened.min(axis=1, keepdims=True)
        min_indices = min_indices.astype(int) # (N, 5) non-unique argmin solution
        bd_pred_arr = min_indices 
        linear_dimensions+=5
    else:
        bd_pred_arr = np.array([])

    return bd_pred_arr, linear_dimensions


def calculate_weights(matches, num_agents):
    # NOTE: calculate weights
    num_agent_goal_ratio = np.mean(matches)
    weights = np.ones(num_agents) # (N,)
    # weights[np.invert(matches.flatten())] = 1/(np.sum(matches)+1)
    weights[matches.flatten()] -= num_agent_goal_ratio+0.001
    # weights[matches.flatten()] = 1/(np.sum(matches)+1)
    # weights += (num_agents-np.sum(weights))/num_agents # TODO this is buggy
    weights *= num_agents/np.sum(weights)
    
    # stay_locations = labels[:,0] == 1

    # random_values = np.random.rand(*stay_locations.shape)
    # flip_mask = (stay_locations == 1) & (random_values > 0.2) # make it so that with 80% probability it gets flipped to 0
    # weights[flip_mask] = 0
    
    return weights


def create_data_object(pos_list, bd_list, grid, priorities, k, m, num_priority_copies, goal_locs, extra_layers, bd_pred, labels=np.array([]), debug_checks=False):
    """
    pos_list: (N,2) positions
    bd_list: (N,W,H) bd's
    grid: (W,H) grid
    priorities: (N,) EECBS priorities
    k: (int) local region size
    m: (int) number of closest neighbors to consider
    # """
    num_layers = 2 # grid and bd_slices intially
    if extra_layers is not None:
        if "agent_locations" in extra_layers:
            num_layers+=1
        if "agent_goal" in extra_layers:
            num_layers+=1
        if "near_goal_info" in extra_layers:
            num_layers+=2
        if "at_goal_grid" in extra_layers:
            num_layers+=1
    
    num_agents = len(pos_list)
    range_num_agents = np.arange(num_agents)

    ### Numpy advanced indexing to get all agent slices at once
    rowLocs = pos_list[:,0][:, None] # (N)->(N,1), Note doing (N)[:,None] adds an extra dimension
    colLocs = pos_list[:,1][:, None] # (N)->(N,1)
    goalRowLocs, goalColLocs = goal_locs[:,0][:, None], goal_locs[:,1][:, None]  # (N,1), (N,1)
    matches = (rowLocs == goalRowLocs) & (colLocs == goalColLocs)
    if debug_checks:
        assert(grid[pos_list[:,0], pos_list[:,1]].all() == 0) # Make sure all agents are on empty space

    # pdb.set_trace()
    x_mesh, y_mesh = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1), indexing='ij') # Each is (D,D)
    # Adjust indices to gather slices
    x_mesh = x_mesh[None, :, :] + rowLocs[:, None, :] # (1,D,D) + (N,1,1) -> (N,D,D)
    y_mesh = y_mesh[None, :, :] + colLocs[:, None, :] # (1,D,D) + (N,1,1) -> (N,D,D)
    grid_slices = grid[x_mesh, y_mesh] # (W, H) -> (N,D,D)
    bd_slices = bd_list[range_num_agents[:,None,None], x_mesh, y_mesh] # (N,D,D)

    node_features = calculate_node_features(bd_slices, grid_slices, rowLocs, colLocs, goalRowLocs, goalColLocs, matches, x_mesh, y_mesh, 
                                      grid, goal_locs, extra_layers, num_layers, num_agents) # (N,num_layers,D,D)
    if debug_checks:
        assert(node_features[:,0,k,k].all() == 0) # Make sure all agents are on empty space
    
    # Edge features include priorities
    edge_indices, edge_features = calculate_edge_features(pos_list, num_agents, range_num_agents, m, k, priorities, num_priority_copies)
    bd_pred_arr, linear_dimensions = calculate_bd_pred_arr(grid_slices, rowLocs, colLocs, num_layers, num_agents, bd_pred)

    weights = calculate_weights(matches, num_agents) # (N,)
    
    return Data(x=torch.from_numpy(node_features), edge_index=torch.from_numpy(edge_indices), 
                edge_attr=torch.from_numpy(edge_features), bd_pred=torch.from_numpy(bd_pred_arr), lin_dim=linear_dimensions, num_channels=num_layers,
                weights = torch.from_numpy(weights), y = torch.from_numpy(labels))


class MyOwnDataset(Dataset):
    def __init__(self, mapNpzFile, bdNpzFolder, pathNpzFolder,
                processedOutputFolder, num_cores, k, m, num_priority_copies, extra_layers, bd_pred, num_per_pt):
        if num_cores != 1:
            raise NotImplementedError("Multiprocessing not supported yet")
        self.extra_layers = extra_layers
        self.bd_pred = bd_pred
        self.mapNpzFile = mapNpzFile
        self.bdNpzFolder = bdNpzFolder
        self.pathNpzFolder = pathNpzFolder
        self.processedOutputFolder = processedOutputFolder
        if not osp.exists(self.processedOutputFolder):
            os.makedirs(self.processedOutputFolder)
        self._raw_file_names = None # Use this to avoid recomputing raw_file_names everytime

        self.ct = CustomTimer()

        self.num_cores = num_cores
        self.num_priority_copies = num_priority_copies
        self.k = k # padding size
        self.m = m # number of agents considered close
        self.size = float('inf')
        self.max_agents = 500
        self.num_per_pt = num_per_pt

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
        pos_list, labels, bd_list, grid, goal_locs, priorities = time_instance

        curdata = create_data_object(pos_list, bd_list, grid, priorities, self.k, self.m, self.num_priority_copies, goal_locs, self.extra_layers, self.bd_pred, labels)
        curdata = apply_masks(len(curdata.x), curdata) # Adds train and test masks to data
        # torch.save(curdata, osp.join(self.processed_dir, f"data_{idx}.pt"))
        return curdata

    def custom_process(self):
        self.load_status_data() # This loads self.df which is used for checking if should process or skip

        if self.pathNpzFolder is not None: # Run process
            # idx = 0
            bd_folder = os.listdir(self.bdNpzFolder)
            print(f"Num cores: {self.num_cores}")
            for npz_path in self.raw_paths: #TODO check if new npzs are read
                # raw_path = "data_collection/data/logs/EXP_Test/iter0/eecbs_npzs/brc202d_paths.npz"
                # if "maps" in raw_path or "bds" in raw_path:
                #     continue
                if not npz_path.endswith(".npz"):
                    continue
                map_name = npz_path.split("/")[-1].removesuffix("_paths.npz")
                # cur_path_iter = raw_path.split("_")[-1][:-4]
                # if not (int(cur_path_iter)==self.iternum):
                #     continue
                df_row = self.df.loc[self.df['npz_path'] == npz_path]
                # assert(len(df_row) <= 1)
                if len(df_row) >0:
                    if df_row.iloc[0]["status"] == "processed":
                        print(f"Skipping: {npz_path}")
                        # idx+=df_row.iloc[0]["num_pts"]
                        continue

                # cur_dataset = data_manipulator.PipelineDataset(raw_path, self.k, self.size, self.max_agents)
                maps_bds = list(filter(lambda k: map_name in k and "bds.npz" == k[-7:], bd_folder))
                idx_start = 0 
                for bd_file_name in maps_bds:
                    bdNpzFile = f"{self.bdNpzFolder}/{bd_file_name}"
                    goalsNpzFile = bdNpzFile[:-7] + "goals.npz"
                    self.ct.start("Loading")
                    cur_dataset = data_manipulator.PipelineDataset(self.mapNpzFile, goalsNpzFile, bdNpzFile, npz_path, self.k, self.size, self.max_agents)
                    print(f"Loading: {npz_path} of size {len(cur_dataset)}")

                    self.ct.stop("Loading")
                    self.ct.start("Processing")
                    # pr = cProfile.Profile()
                    # pr.enable()
                    # tmp = []
                    counter = 0
                    batch_graphs = []
                    for t in tqdm(range(len(cur_dataset))):
                        time_instance = cur_dataset[t]
                        torch.save(self.create_and_save_graph(t, time_instance),
                                    osp.join(self.processed_dir, f"data_{map_name}_{idx_start+t}.pt"))
                    
                    idx_start+=len(cur_dataset)
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
                    if(len(cur_dataset)>0):
                        new_df = pd.DataFrame.from_dict({"npz_path": [npz_path],
                                                        "pt_path": [f"data_{map_name}"],
                                                        "status": ["processed"], 
                                                        "num_pts": [len(cur_dataset)],
                                                        "loading_time": [self.ct.getTimes("Loading", "list")[-1]], 
                                                        "pxrocessing_time": [self.ct.getTimes("Processing", "list")[-1]]})
                        if len(self.df) == 0:
                            self.df = new_df
                        else:
                            self.df = pd.concat([self.df, new_df], ignore_index=True)
                        self.df.to_csv(self.df_path, index=False)
                    
                    del cur_dataset
            # self.length = idx
        
        self.length = self.df["num_pts"].sum()
        print("Loading dataset with length: ", self.length)

        ### Get indices and files
        # self.order_of_indices looks like [0, 213, 457, 990, 1405, ...]
        # self.order_of_files looks like [map1, map2, map2, map4, map5, ...]
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

        # for row in self.df.iterrows():
        #     self.order_to_loaded_pt.append(torch.load(osp.join(self.processed_dir, row[1]["pt_path"])))
        
    def len(self):
        """Require to override Dataset len()"""
        return self.length

    def get(self, idx):
        """Require to override Dataset get()"""
        which_file_index = np.searchsorted(self.order_of_indices, idx, side='right')-1 # (num_maps)
        # data_file = self.order_of_files[which_file_index]
        data_idx = idx-self.order_of_indices[which_file_index]
        assert(data_idx >= 0)
        filename = f"{self.order_of_files[which_file_index]}_{data_idx}.pt"
        curdata = torch.load(osp.join(self.processed_dir, filename))

        return normalize_graph_data(curdata, self.k, edge_normalize="k", bd_normalize="center")


### Example run
"""
python -m gnn.dataloader --mapNpzFile=data_collection/data/benchmark_data/constant_npzs/all_maps.npz \
      --bdNpzFolder=data_collection/data/benchmark_data/constant_npzs \
      --pathNpzFolder=data_collection/data/logs/EXP_Test_batch/iter0/eecbs_npzs \
      --processedFolder=data_collection/data/logs/EXP_Test_batch/iter0/processed \
      --k=5 \
      --m=3 \
      --num_priority_copies=10
"""

if __name__ == "__main__":
    """
    This file takes in npzs and processes them into pt files.

    We assume the following file structure:
    Inputs:
    - mapNpzFile: e.g. all_maps.npz with all_maps[MAPNAME] = grid_map (H,W)
    - bdNpzFolder: Folder containing [MAPNAME]_bds.npz and [MAPNAME]_goals.npz  
    - pathNpzFolder: Folder containing [MAPNAME]_paths.npz
    Outputs:
    - processedFolder: Folder to save processed data, will contain [MAPNAME]_[idx].pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapNpzFile", help="map npz file", type=str, required=True)
    parser.add_argument("--bdNpzFolder", help="bd and goals npz folder", type=str, required=True)
    parser.add_argument("--pathNpzFolder", help="path npz folder", type=str, required=True)
    parser.add_argument("--processedFolder", help="processed folder to save pt", type=str, required=True)
    parser.add_argument("--k", help="window size", type=int)
    parser.add_argument("--m", help="num_nearby_agents", type=int)
    parser.add_argument("--num_priority_copies", help="copies of relative priority to include in input", type=int, default=1)
    extraLayersHelp = "Types of additional layers for training, comma separated. Options are: agent_locations, agent_goal, at_goal_grid"
    parser.add_argument('--extra_layers', help=extraLayersHelp, type=str, default=None)
    parser.add_argument('--bd_pred', type=str, default=None, help="bd_predictions added to NN, type anything if adding")
    parser.add_argument('--num_per_pt', type=int, default=16, help="number of graphs per pt file")
    args = parser.parse_args()

    dataset = MyOwnDataset(mapNpzFile=args.mapNpzFile, bdNpzFolder=args.bdNpzFolder, 
                        pathNpzFolder=args.pathNpzFolder, processedOutputFolder=args.processedFolder,
                        num_cores=1, k=args.k, m=args.m, num_priority_copies=args.num_priority_copies, extra_layers=args.extra_layers, bd_pred=args.bd_pred, num_per_pt=args.num_per_pt)













# NOTE code to check if bd_preds are correct
    # if bd_predictions:
    #     best_moves2 = np.zeros((num_agents, 5))
    #     _,row, col = bd_slices.shape
    #     for i, bd in enumerate(bd_slices):
    #         stay = bd[int(row/2), int(col/2)]
    #         right = bd[int(row/2), int(col/2)+1]
    #         left = bd[int(row/2), int(col/2)-1]
    #         up = bd[int(row/2)-1, int(col/2)]
    #         down = bd[int(row/2)+1, int(col/2)]
    #         act_list =[stay, right, down, up, left]
    #         min_act = min(act_list)
    #         if stay == min_act:
    #             best_moves2[i,0] = 1
    #         if right == min_act:
    #             best_moves2[i,1] = 1
    #         if down == min_act:
    #             best_moves2[i,2] = 1
    #         if up == min_act:
    #             best_moves2[i,3] = 1
    #         if left == min_act:
    #             best_moves2[i,4] = 1    
    # assert(np.all(bd_pred_arr==best_moves2))
    
    
    
    # def process_single(self, npz_path):
    #     """
    #     Still in progress, do not use!
    #     """
    #     shared_lock = self.shared_lock
    #     assert(npz_path.endswith(".npz"))
    #     map_name = npz_path.split("/")[-1].removesuffix("_paths.npz")
    #     shared_lock.acquire()
    #     df_row = self.df.loc[self.df['npz_path'] == npz_path]
    #     assert(len(df_row) <= 1)
    #     if len(df_row) == 1:
    #         if df_row.iloc[0]["status"] == "processed":
    #             print(f"Skipping: {npz_path}")
    #         else:
    #             print(f"WARNING: Unexpected status for {map_name}: {df_row.iloc[0]['status']}")
    #     shared_lock.release()

    #     bdNpzFile = f"{self.bdNpzFolder}/{map_name}_bds.npz"
    #     cur_dataset = data_manipulator.PipelineDataset(self.mapNpzFile, bdNpzFile, npz_path, self.k, self.size, self.max_agents)
    #     print(f"Loading: {npz_path} of size {len(cur_dataset)}")

    #     tmp = []
    #     for t in tqdm(range(len(cur_dataset))):
    #         time_instance = cur_dataset[t]
    #         tmp.append(self.create_and_save_graph(idx+t, time_instance))
    #     torch.save(tmp, osp.join(self.processed_dir, f"data_{map_name}.pt"))
    #     idx += len(cur_dataset)

    #     new_df = pd.DataFrame.from_dict({"npz_path": [npz_path],
    #                                     "pt_path": [f"data_{map_name}.pt"],
    #                                     "status": ["processed"], 
    #                                     "num_pts": [len(cur_dataset)],
    #                                     "loading_time": [self.ct.getTimes("Loading", "list")[-1]], 
    #                                     "processing_time": [self.ct.getTimes("Processing", "list")[-1]]})
        
    #     shared_lock.acquire()
    #     if len(self.df) == 0:
    #         self.df = new_df
    #     else:
    #         self.df = pd.concat([self.df, new_df], ignore_index=True)
    #     self.df.to_csv(self.df_path, index=False)
    #     shared_lock.release()
    #     raise NotImplementedError("Not implemented yet")