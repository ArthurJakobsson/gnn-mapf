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

## Repository Structure
### Data Collection
- Manages the data collection process by calling the simulator and eecbs in parallel.
- Stores collected data.

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


