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
conda env create -f environment.yml
```
This creates a conda environment named `ssil` that you should use.

To download data assets, i.e., maps, scenes, and pretrained model
```sh 
bash download_assets.bash
```
Note: gdown in the above command might complain at some point due to data limits. If so, wait a few minutes and rerun the command.

To run our pre-trained model on a map, look at the `simulator.py` file. Here is an example command:
```sh
python -m simulator.py ...
```

To visualize outputs, use:
```sh

```

If you are interested in running your own large scale data collection, please take a look at the "full-framework" branch. Note that the branch is not clean and cannot be run out of the box, but we hope it can still help if someone is interested in large scale data collection for training MAPF models.


## Citation
If you use this repository in your research, please cite our work:

```bibtex
@article{veerapaneni2024worksmarterhardersimple,
      title={Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large Scale Imitation Learning for MAPF}, 
      author={Rishi Veerapaneni and Arthur Jakobsson and Kevin Ren and Samuel Kim and Jiaoyang Li and Maxim Likhachev},
      year={2024},
      eprint={2409.14491},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2409.14491}, 
}
```


## Development Status
This project is semi-actively being developed. For questions or contributions, contact: **rveerapa@andrew.cmu.edu** or **ajakobss@cmu.edu** 
