.. data_collection:

Data Collection
================

eecbs_batchrunner.py
--------------------

This module provides a generic batch runner for Multi-Agent Path Finding (MAPF) solvers,
specifically EECBS and Python-based ML models.

The batch runner handles:

#. Parallel execution of MAPF solver runs across multiple scenarios and maps
#. Data collection and processing of results
#. Conversion of results to NPZ format for machine learning applications

.. note::

   Some of the below documentation has been created with the assistance of generative AI and so should be taken with a grain of salt.

Module Constants
^^^^^^^^^^^^^^^^^

.. data:: mapsToMaxNumAgents

   Dictionary mapping map names to maximum number of agents each map can handle

Tmux Session Functions
^^^^^^^^^^^^^^^

.. py:function:: createTmuxSession(i)

   Create a new tmux session with a given index.
   
   :param int i: Index for the tmux session

.. py:function:: runCommandWithTmux(i, command)

   Run a command in a tmux session with a given index.
   
   :param int i: Index of the tmux session
   :param str command: Command to run in the tmux session

.. py:function:: killTmuxSession(i)

   Kill a tmux session with a given index.
   
   :param int i: Index of the tmux session to kill

Command Generation
^^^^^^^^^^^^^^^^^^^

.. py:function:: getEECBSCommand(eecbsArgs, outputFolder, outputfile, mapfile, numAgents, scenfile)

   Generate the command for running EECBS.
   
   :param dict eecbsArgs: Arguments for EECBS
   :param str outputFolder: Folder for output files
   :param str outputfile: File for EECBS output
   :param str mapfile: Path to map file
   :param int numAgents: Number of agents
   :param str scenfile: Path to scenario file
   :return: Command for running EECBS
   :rtype: str
    
.. py:function:: getPyModelCommand(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile)

   Generate the command for running the Python ML model.
   
   :param dict runnerArgs: Arguments for the Python model
   :param str outputFolder: Folder for output files
   :param str outputfile: File for model output
   :param str mapfile: Path to map file
   :param int numAgents: Number of agents
   :param str scenfile: Path to scenario file
   :return: Command for running the Python model
   :rtype: str

.. py:function:: getCommandForSingleInstance(runnerArgs, outputFolder, outputfile, mapfile, numAgents, scenfile)

   Get the command for running a single instance based on the runner type.
   
   :param dict runnerArgs: Arguments for the runner
   :param str outputFolder: Folder for output files
   :param str outputfile: File for output
   :param str mapfile: Path to map file
   :param int numAgents: Number of agents
   :param str scenfile: Path to scenario file
   :return: Command for running the instance
   :rtype: str
   :raises: ValueError if the command is unknown

Status Detection
^^^^^^^^^^^^^^^^

.. py:function:: detectExistingStatus(runnerArgs, mapfile, aNum, scenfile, df)

   Detect if the current configuration has already been run and if it was successful.
   
   :param dict runnerArgs: Arguments for the runner
   :param str mapfile: Path to map file
   :param int aNum: Number of agents
   :param str scenfile: Path to scenario file
   :param df: DataFrame or path to CSV file with results
   :type df: pandas.DataFrame or str
   :return: Tuple of (has_been_run, success_status)
   :rtype: tuple
   :raises: KeyError if a key is not found in the dataframe or the command is unknown

Multi-threaded Execution
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: runSingleInstanceMT(queue, nameToNumRun, lock, worker_id, idToWorkerOutputFilepath, static_dict, runnerArgs, mapName, curAgentNum, scen)

   Run a single instance of the MAPF solver in multi^threading mode.
   
   :param multiprocessing.Queue queue: Queue for communication between processes
   :param dict nameToNumRun: Dictionary mapping map names to number of remaining runs
   :param multiprocessing.Lock lock: Lock for thread safety
   :param int worker_id: ID of the worker process
   :param callable idToWorkerOutputFilepath: Function to get the output file path
   :param dict static_dict: Dictionary with static information
   :param dict runnerArgs: Arguments for the runner
   :param str mapName: Name of the map
   :param int curAgentNum: Number of agents
   :param str scen: Path to scenario file

.. py:function:: checkIfRunNextAgents(queue, nameToNumRun, lock, num_workers, idToWorkerOutputFilepath, static_dict, eecbsArgs, mapName, curAgentNum)

   Check if the next agent numbers should be run after completing all runs for the current agent number.
   
   :param multiprocessing.Queue queue: Queue for communication between processes
   :param dict nameToNumRun: Dictionary mapping map names to number of remaining runs
   :param multiprocessing.Lock lock: Lock for thread safety
   :param int num_workers: Number of worker processes
   :param callable idToWorkerOutputFilepath: Function to get the output file path
   :param dict static_dict: Dictionary with static information
   :param dict eecbsArgs: Arguments for EECBS
   :param str mapName: Name of the map
   :param int curAgentNum: Current number of agents

.. py:function:: worker(queue, nameToNumRun, lock, worker_id, num_workers, static_dict, idToWorkerOutputFilepath)

   Worker process function that processes tasks from the queue.
   
   :param multiprocessing.JoinableQueue queue: Queue for communication between processes
   :param dict nameToNumRun: Dictionary mapping map names to number of remaining runs
   :param multiprocessing.Lock lock: Lock for thread safety
   :param int worker_id: ID of the worker process
   :param int num_workers: Number of worker processes
   :param dict static_dict: Dictionary with static information
   :param callable idToWorkerOutputFilepath: Function to get the output file path
   :raises: ValueError if the function is unknown

.. py:function:: helperRun(command)

   Helper function to run a command in a shell.
   
   :param str command: Command to run

Setup and Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: specificRunnerDictSetup(args)

   Set up the runner arguments dictionary based on command type.
   
   :param argparse.Namespace args: Command line arguments
   :return: Runner arguments dictionary
   :rtype: dict
   :raises: ValueError if the command is unknown

.. py:function:: eecbs_runner_setup(args)

   Set up the global variables and paths for EECBS runner.
   
   :param argparse.Namespace args: Command line arguments

.. py:function:: runDataManipulator(args, ct, mapsToScens, static_dict, outputPathNpzFolder, mapsInputFolder, num_workers)

   Run the data manipulator to convert outputs to NPZ format.
   
   :param argparse.Namespace args: Command line arguments
   :param CustomTimer ct: Timer object for measuring execution time
   :param dict mapsToScens: Dictionary mapping map names to scenario files
   :param dict static_dict: Dictionary with static information
   :param str outputPathNpzFolder: Folder for output NPZ files
   :param str mapsInputFolder: Folder with input map files
   :param int num_workers: Number of worker processes

.. py:function:: generic_batch_runner(args)

   Main function for the generic batch runner.
   
   This function handles the overall execution flow, including:
   
   ^ Setting up the filesystem
   ^ Starting worker processes
   ^ Creating jobs
   ^ Processing results
   ^ Running the data manipulator
   
   :param argparse.Namespace args: Command line arguments

Usage Examples
^^^^^^^^^^^^^^

Basic usage with EECBS:

.. code-block:: bash

    python -m data_collection.eecbs_batchrunner 
      --mapFolder=data_collection/data/benchmark_data/maps \
      --scenFolder=data_collection/data/benchmark_data/scens \
      --constantMapAndBDFolder=data_collection/data/benchmark_data/constant_npzs2 \
      --outputFolder=data_collection/data/logs/EXP_Test_batch/iter0/eecbs_outputs \
      --num_parallel_runs=50 \
      "eecbs" \
      --outputPathNpzFolder=data_collection/data/logs/EXP_Test_batch/iter0/eecbs_npzs \
      --firstIter=false --cutoffTime=5

Basic usage with Python model:

.. code-block:: bash

    python -m data_collection.eecbs_batchrunner 
      --mapFolder=data_collection/data/benchmark_data/maps \
      --scenFolder=data_collection/data/benchmark_data/scens \
      --constantMapAndBDFolder=data_collection/data/benchmark_data/constant_npzs2 \
      --outputFolder=data_collection/data/logs/EXP_Test_batch/iter0/pymodel_outputs \
      --num_parallel_runs=50 \
      "pymodel" \
      --modelPath=data_collection/data/logs/EXP_Test2/iter0/models/max_test_acc.pt \
      --k=4 --m=5 --maxSteps=100 --shieldType=CS-PIBT

data_manipulator.py
-------------------

This file processes raw data from EECBS solver runs and converts them into NPZ format
suitable for machine learning applications. It handles maps, backward Dijkstra (BD) values,
and path data.

Key operations:

* Parse map files (.map) to NumPy arrays
* Process backward Dijkstra (BD) files to NumPy arrays
* Convert agent path information to NumPy arrays
* Save data in compressed NPZ format for efficient loading

Classes
^^^^^^^

.. class:: PipelineDataset(Dataset)

   A PyTorch Dataset for loading EECBS instances for training ML models.

   .. method:: __init__(self, mapFileNpz, bdFileNpz, pathFileNpz, k, size, max_agents, helper_bd_preprocess="middle")
      
      :param str mapFileNpz: Path to NPZ file containing map data
      :param str bdFileNpz: Path to NPZ file containing backward Dijkstra data
      :param str pathFileNpz: Path to NPZ file containing path data
      :param int k: Window size for local observation
      :param int size: Maximum size of dataset
      :param int max_agents: Maximum number of agents
      :param str helper_bd_preprocess: Method to center helper backward Dijkstras ('middle', 'current', or 'subtraction')

   .. method:: __len__(self)
      
      :return: Number of instances in the dataset
      :rtype: int

   .. method:: __getitem__(self, idx)
      
      Retrieves an item from the dataset, providing the local observation window.
      
      :param int idx: Index of the instance to retrieve
      :return: Tuple of (current_locations, one_hot_labels, backward_dijkstra, grid_map, goal_locations)
      :rtype: tuple

   .. method:: find_instance(self, idx)
      
      Finds the specific instance based on the index.
      
      :param int idx: Index to find
      :return: Tuple of (backward_dijkstra, grid_map, paths, timestep, max_timesteps)
      :rtype: tuple

   .. method:: parse_npz(self, loaded_paths, loaded_maps, loaded_bds)
      
      Parses loaded NPZ data and prepares it for dataset access.
      
      :param dict loaded_paths: Dictionary of path data
      :param dict loaded_maps: Dictionary of map data
      :param dict loaded_bds: Dictionary of backward Dijkstra data

   .. method:: parse_npz2(self)
      
      Alternative parsing method that filters and validates data.

File Parsing Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. function:: parse_map(mapfile)
   
   Parses a map file into a NumPy array.
   
   :param str mapfile: Path to map file
   :return: 2D array where 1 represents obstacles and 0 represents free space
   :rtype: numpy.ndarray

.. function:: parse_path(pathfile)
   
   Parses a path file containing agent movements over time.
   
   :param str pathfile: Path to path file
   :return: 3D array of shape (timesteps, num_agents, 2) containing agent positions
   :rtype: numpy.ndarray

.. function:: parse_bd(bdfile)
   
   Parses a backward Dijkstra file into a NumPy array.
   
   :param str bdfile: Path to backward Dijkstra file
   :return: 3D array of shape (num_agents, height, width) containing distance values
   :rtype: numpy.ndarray

Batch Processing Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: batch_map(dir, num_parallel)
   
   Processes multiple map files in parallel.
   
   :param str dir: Directory containing map files
   :param int num_parallel: Number of parallel processes to use
   :return: Dictionary mapping filenames to map arrays
   :rtype: dict

.. function:: batch_bd(dir, num_parallel)
   
   Processes multiple backward Dijkstra files in parallel.
   
   :param str dir: Directory containing BD files
   :param int num_parallel: Number of parallel processes to use
   :return: Dictionary mapping scenario names to BD arrays
   :rtype: dict

.. function:: batch_path(dir)
   
   Processes multiple path files.
   
   :param str dir: Directory containing path files
   :return: Dictionary mapping key strings to path arrays
   :rtype: dict

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: main()
   
   Entry point for the command-line interface. Parses arguments and orchestrates
   the processing of maps, backward Dijkstra values, and paths.

Usage Examples
^^^^^^^^^^^^^^

Example usage from command line:

.. code-block:: bash

    python -m data_collection.data_manipulator 
      --pathsIn=data_collection/data/logs/EXP_Collect_BD/iter0/eecbs_outputs/empty_8_8/paths/ 
      --pathOutFile=data_collection/data/logs/EXP_Collect_BD/iter0/eecbs_npzs/empty_8_8_paths.npz 
      --bdIn=data_collection/data/logs/EXP_Collect_BD/iter0/eecbs_outputs/empty_8_8/bd 
      --bdOutFile=data_collection/data/benchmark_data/constant_npzs2/empty_8_8_bds.npz 
      --mapIn=data_collection/data/benchmark_data/maps 
      --mapOutFile=data_collection/data/benchmark_data/constant_npzs2/empty_8_8_map.npz 
      --num_parallel=1