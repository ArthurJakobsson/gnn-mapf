import argparse
import subprocess
import pandas as pd
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import gc
from multiprocessing import Pool
import pdb
from itertools import repeat
import time
import torch_geometric.inspector

agent_counts = [8,16,32,64,128]

for agent_num in agent_counts:
    command = f" python -m benchmarking.benchmark_runner --mapname=all --numAgents=increment --data_folder=data_collection/data/benchmark_original --map_folder=data_collection/data/benchmark_original/maps --scen_folder=data_collection/data/benchmark_original/scens --extra_layers=agent_locations --model_path=data_collection/data/logs/multi_scen_runs/{agent_num}/models/max_test_acc.pt --num_parallel=64 --bd_pred=t --eecbs_cutoff=120 --simulator_cutoff=120 --pymodel_out=benchmarking/{agent_num}_agents_results"
    print(command)
    subprocess.run(command, shell=True, check=True)