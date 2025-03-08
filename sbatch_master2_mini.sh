#! /bin/bash

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2

# python sbatch_master_process_runner2.py --machine=psc \
#     --collect_data \
#     --data_path=mini_benchmark_data --exp_path=EXP_mini_0_1 # --clean
python sbatch_master_process_runner2.py --machine=psc \
    --collect_data \
    --data_path=mini_benchmark_data --exp_path=EXP_mini_3_1 # --clean

# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --models="ResGatedGraphConv" --use_edge_attr \
#     --data_path=mini_benchmark_data --exp_path=EXP_mini_0_1 # --logging --clean
python sbatch_master_process_runner2.py --machine=psc \
    --trainer --models="ResGatedGraphConv" --use_edge_attr \
    --data_path=mini_benchmark_data --exp_path=EXP_mini_3_1 # --logging --clean