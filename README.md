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

To install dependencies, run:
```sh
conda config --set channel_priority flexible
conda env create -f pytorchfun.yml
conda install gdown
```

To download data assets, i.e., maps, scenes, and pretrained model
```sh 
bash download_assets.bash
```
Note: gdown in the above command might complain at some point due to data limits. If so, comment out the completed lines, wait, and restart.

To run our pre-trained model on a map, use:
```sh
python simulator.py ...
```

To visualize outputs, use:
```sh

```

If you are interested in running your own large scale data collection, please take a look at the "full-framework" branch. Note that the branch is not clean and cannot be run out of the box, but we hope it can still help if someone is interested in large scale data collection for training MAPF models.


## Development Status
This project is semi-actively being developed. For questions or contributions, contact: **rveerapa@andrew.cmu.edu** or **ajakobss@cmu.edu** 


