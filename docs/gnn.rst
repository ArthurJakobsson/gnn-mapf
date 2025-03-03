GNN
===

.. note::

   Some of the below documentation has been created with the assistance of generative AI and so should be taken with a grain of salt.

dataloader.py
-------------

This file converts map, backward Dijkstra, and path data into graph-structured data for
training Graph Neural Networks (GNNs). It handles the creation of node features, edge connections,
and labels for multi-agent path finding tasks.

Key operations:

* Convert raw MAPF data into PyTorch Geometric Data objects
* Generate local observation windows for each agent
* Create graph structures with appropriate node and edge features
* Process and normalize features for neural network training
* Saves graphs to folder for use 

Utility Functions
^^^^^^^^^^^^^^^^

apply_masks
^^^^^^^^^^^

.. function:: apply_masks(data_len, curdata)
   
   Applies train/test masks to split data for training and evaluation.
   
   :param int data_len: Length of the data
   :param Data curdata: PyTorch Geometric Data object
   :return: Data object with train_mask and test_mask attributes
   :rtype: Data

normalize_graph_data
^^^^^^^^^^^^^^^^^^^^

.. function:: normalize_graph_data(data, k, edge_normalize="k", bd_normalize="center")
   
   Normalizes edge attributes and backward Dijkstra values in the graph data.
   
   :param Data data: PyTorch Geometric Data object
   :param int k: Window size for local observation
   :param str edge_normalize: Method for normalizing edge attributes
   :param str bd_normalize: Method for normalizing backward Dijkstra values
   :return: Normalized Data object
   :rtype: Data

get_bd_prefs
^^^^^^^^^^^^

.. function:: get_bd_prefs(pos_list, bds, range_num_agents)
   
   Gets movement preferences based on backward Dijkstra values.
   
   :param numpy.ndarray pos_list: Agent positions of shape (N,2)
   :param numpy.ndarray bds: Backward Dijkstra values of shape (N,W,H) 
   :param numpy.ndarray range_num_agents: Range of agent indices
   :return: Agent movement preferences
   :rtype: numpy.ndarray

create_data_object
^^^^^^^^^^^^^^^^^^

.. function:: create_data_object(pos_list, bd_list, grid, k, m, goal_locs, extra_layers, bd_pred, labels=np.array([]), debug_checks=False)
   
   Creates a PyTorch Geometric Data object from raw MAPF data. The usable extra information is agent_locations, agent_goal, near_goal_info, and at_goal_grid as well as bds which are almost always used.
   Agent locations provides the locations of the agents nearby, agent goal provides the goal locations of the agents nearby, near_goal_info provides the agents near the goal, and at_goal_grid provides a binary grid of the agents if they are at their goal.
   
   :param numpy.ndarray pos_list: Agent positions of shape (N,2)
   :param numpy.ndarray bd_list: Backward Dijkstra values of shape (N,W,H)
   :param numpy.ndarray grid: Grid map of shape (W,H)
   :param int k: Window size for local observation
   :param int m: Number of closest neighbors to consider
   :param numpy.ndarray goal_locs: Goal locations of shape (N,2)
   :param list extra_layers: List of additional features to include
   :param str bd_pred: Whether to include backward Dijkstra predictions
   :param numpy.ndarray labels: Ground truth labels
   :param bool debug_checks: Whether to perform debug assertions
   :return: PyTorch Geometric Data object
   :rtype: Data

Dataset Classes
^^^^^^^^^^^^^^

MyOwnDataset
^^^^^^^^^^^^

.. class:: MyOwnDataset(Dataset)

   A PyTorch Geometric Dataset for processing and loading MAPF data as graphs.

   .. method:: __init__(self, mapNpzFile, bdNpzFolder, pathNpzFolder, processedOutputFolder, num_cores, k, m, extra_layers, bd_pred, num_per_pt)
      
      :param str mapNpzFile: Path to NPZ file containing map data
      :param str bdNpzFolder: Path to folder containing backward Dijkstra NPZ files
      :param str pathNpzFolder: Path to folder containing path NPZ files
      :param str processedOutputFolder: Path to output folder for processed data
      :param int num_cores: Number of CPU cores to use for parallel processing
      :param int k: Window size for local observation
      :param int m: Number of closest neighbors to consider
      :param list extra_layers: List of additional features to include
      :param str bd_pred: Whether to include backward Dijkstra predictions
      :param int num_per_pt: Number of graphs per PT file

   .. method:: processed_dir(self) -> str
      
      :return: Path to the processed directory
      :rtype: str

   .. method:: load_status_data(self)
      
      Loads a CSV file tracking the status of processed files.

   .. method:: raw_file_names(self)
      
      :return: List of raw file paths
      :rtype: list

   .. method:: has_process(self) -> bool
      
      :return: Whether the dataset has a process method
      :rtype: bool

   .. method:: has_download(self) -> bool
      
      :return: Whether the dataset has a download method
      :rtype: bool

   .. method:: create_and_save_graph(self, idx, time_instance)
      
      Creates and saves a graph from a time instance.
      
      :param int idx: Index of the graph
      :param tuple time_instance: Tuple of (pos_list, labels, bd_list, grid, goal_locs)
      :return: PyTorch Geometric Data object
      :rtype: Data

   .. method:: custom_process(self)
      
      Process all data files and tracks processing status.

   .. method:: len(self)
      
      :return: Number of graphs in the dataset
      :rtype: int

   .. method:: get(self, idx)
      
      Retrieves a graph by index.
      
      :param int idx: Index of the graph to retrieve
      :return: PyTorch Geometric Data object
      :rtype: Data

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

The dataloader can be executed as a command-line tool:

.. code-block:: bash

    python -m gnn.dataloader 
      --mapNpzFile=data_collection/data/benchmark_data/constant_npzs/all_maps.npz 
      --bdNpzFolder=data_collection/data/benchmark_data/constant_npzs 
      --pathNpzFolder=data_collection/data/logs/EXP_Test3/iter0/eecbs_npzs 
      --processedFolder=data_collection/data/logs/EXP_Test3/iter0/processed 
      --k=4 
      --m=5 
      --extra_layers=agent_locations,agent_goal,at_goal_grid 
      --bd_pred=1

trainer.py
-----------

This file trains a GNN model on MAPF data using PyTorch Geometric. It uses a SageConv model to predict agent actions.

There are some *hyperparameter* modifications that can be done to training. This includes k, m, bd_pred, and extra_layers. The k and m are the same as in the dataloader. The bd_pred is whether to include backward Dijkstra predictions. The extra_layers are the extra features to include in the model. Note that changing these may cause recreation of the pt files which is a time and space intensive process.

Command Line Interface

The trainer can be called from the command-line:

.. code-block:: bash

    python -m gnn.trainer --exp_folder=data_collection/data/logs/EXP_Test --experiment=exp0 --iternum=0 --num_cores=4
        --processedFolders=data_collection/data/logs/EXP_Test3/iter0/processed 
        --mapNpzFile=data_collection/data/benchmark_data/constant_npzs/all_maps.npz
        --bdNpzFolder=data_collection/data/benchmark_data/constant_npzs
        --pathNpzFolders=data_collection/data/logs/EXP_Test/iter0/eecbs_npzs