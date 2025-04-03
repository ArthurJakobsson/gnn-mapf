# Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large Scale Imitation Learning for MAPF

This codebase is mainly for archival / documentation purposes. This codebase is unfortunately not directly runable, but hopefully can help others interested in large scale imitation learning for MAPF.
If you would like to run our model (trained on the largest dataset we collected for the paper), please see the [https://github.com/Rishi-V/ML-MAPF-with-Search](https://github.com/Rishi-V/ML-MAPF-with-Search) repo which contains directly runnable code of the model, CS-PIBT, and other collision shields.

Our project website is available at [SSIL Website](https://arthurjakobsson.github.io/ssil_mapf).


## Repository Structure
This repository contains the implementation of a Graph Neural Network (GNN) for Multi-Agent Path Finding (MAPF) using Simulation and DAgger (Dataset Aggregation). The project includes data collection, benchmarking, model training, and simulation components.
### Data Collection
- The `data_collection` folder manages the data collection process by calling the simulator and eecbs in parallel.
- `data_collection/eecbs_batchrunner.py` runs many instances of eecbs in parallel. It takes in a folder with `.scen` and `.map` files (which eecbs takes as inputs) and creates many `.txt` solution files. It uses a semi-hacky system of creating and calling tmux sessions in parallel and should be changed to something like [Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html).
- `data_collection/data_manipulator.py` parses the created `.txt` files into `.npz` files.

### Training
- The `gnn` folder contains the training code.
- `gnn/dataloader.py` parses the created solution path `.npz` files along with the correponding map and backward dijkstra files to create a dataset of `.pt` files, where each `.pt` file correponds to a single agent graph with inputs and target labels. Thus, dataloader handles the processing of creating the local field of view, finding neighbors, and creating the input observations more generally.
- `gnn/trainer.py` trains the model based on the given dataset and model architecture.

### Evaluating / DAgger collection
- The `gnn` folder also contains the evaluation code.
- `gnn/simulator.py` runs the trained model on a given scen and map. It also has the capability to create new scens based on the agents path for DAgger (e.g., if it fails, it could create `.scen` files for the last 10 timesteps to be fed into eecbs afterwards to obtain labels).

## Other Comments
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

### Understanding Data Structure
- Data is stored in `.npz` files containing maps, bd values, and paths.
- Paths are split into multiple files for efficient loading.
- Training data is converted into `.pt` files, each representing a single training instance.
- Benchmark datasets are organized into folders for easy toggling.

### Citation
If this work is relevant to your project, please cite us:
 ```bibtex
@article{veerapaneni2024work_smart_not_harder,
  title = {Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large Scale Imitation Learning for MAPF},
  author = {Veerapaneni, Rishi and Jakobsson, Arthur and Ren, Kevin and Kim, Samuel and Li, Jiaoyang and Likhachev, Maxim},
  year = {2024},
  journal = {arXiv preprint arxiv:2409.14491},
  eprint = {2409.14491},
  archiveprefix = {arXiv},
  primaryclass = {cs.MA},
}
```
