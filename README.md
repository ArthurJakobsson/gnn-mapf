# Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large Scale Imitation Learning for MAPF

For a more comprehensive and fuller explanation please see our documentation website [SSIL GNN MAPF Documentation](https://arthurjakobsson.github.io/ssil_documentation)

Our project website is also available [SSIL Website](https://arthurjakobsson.github.io/ssil_mapf)

## Overview
This repository contains the implementation of a Graph Neural Network (GNN) for Multi-Agent Path Finding (MAPF) using Simulation and DAgger (Dataset Aggregation). The project includes data collection, benchmarking, model training, and simulation components.

## Installation
To clone the repository, run:
```sh
git clone https://github.com/ArthurJakobsson/gnn-mapf.git
```

For active development, use the `GNN-development2` branch.

To install dependencies, run:
```sh
pip install -r requirements.txt
```

To install the data and our largest model, run:

install via pip:
```sh
pip install gdown
```

use it in python:
```py
import gdown
gdown.download_folder(url, quiet=True)
```

where the url is [https://drive.google.com/drive/folders/15G5mmBh5FDEpGlNKAE_gA5je8xV_NYKf?usp=drive_link](https://drive.google.com/drive/folders/15G5mmBh5FDEpGlNKAE_gA5je8xV_NYKf?usp=drive_link)

Please note that this folder is about 40 GB.

## Repository Structure
### Data Collection
- Manages the data collection process by calling the simulator and eecbs in parallel.
- Stores collected data.`

To run data collection OR simulation use the eecbs_batchrunner. Below is an example command where the inputs can be varied depending on file paths and whether simulation, eecbs or some other subprocess is required.

```sh
python -m data_collection.eecbs_batchrunner --mapFolder=./data_collection/data/benchmark_new_maps/maps --scenFolder=./data_collection/data/benchmark_new_maps/scens --numAgents=50 --constantMapAndBDFolder=data_collection/data/benchmark_new_maps/constant_npzs --outputFolder=data_collection/data/logs/EXP_test_agents/iter0/eecbs_outputs --num_parallel_runs=1 "eecbs" --outputPathNpzFolder=data_collection/data/logs/EXP_test_agents/iter0/eecbs_npzs --firstIter=true --cutoffTime=60 --suboptimality=2
```

To run the pymodel (simulation) instead of eecbs you should replace "eecbs" with "pymodel" and provide the information required by the argparse under the pymodel section in eecbs_batchrunner such as model path. Your model can be stored anywhere as long as a correct file path is provided.

### Slurm Implementation
- Facilitates large-scale testing on a cluster.
- Runs batch jobs using `.sh` scripts to execute the next steps automatically.
- Includes checkpointing for easy resumption.

### Benchmarking
- Provides tools to compare model performance against benchmarks such as EPH.
- Includes visualization and plotting scripts.

### GNN
- Contains the core neural network model, training loop, data processing, and evaluation scripts.
- Handles dataset splitting and training instance indexing.

### Simulator
- Located at `gnn/simulator.py`, used for DAgger training, benchmarking, and integration with CS-PIBT.

### New Map Generation
- Generates custom maps and scenarios for experiments.
- Expands the MovingAI Benchmark dataset with additional scenarios.

## Understanding Data Structure
- Data is stored in `.npz` files containing maps, bd values, and paths.
- Paths are split into multiple files for efficient loading.
- Training data is converted into `.pt` files, each representing a single training instance.
- Benchmark datasets are organized into folders for easy toggling.

## Development Status
This project is actively being developed. For questions or contributions, contact: **rveerapa@andrew.cmu.edu** or **ajakobss@cmu.edu** 


