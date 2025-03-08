#! /bin/bash

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2

python sbatch_master_process_runner2.py --machine=psc \
    --collect_data \
    --data_path=benchmark_data --exp_path=EXP_full_0_1 \
    --clean
python sbatch_master_process_runner2.py --machine=psc \
    --collect_data \
    --data_path=benchmark_data --exp_path=EXP_full_3_1 \
    --clean

python sbatch_master_process_runner2.py --machine=psc \
    --trainer --models="ResGatedGraphConv" --use_edge_attr \
    --data_path=benchmark_data --exp_path=EXP_full_0_1 \
    --clean # --logging    
python sbatch_master_process_runner2.py --machine=psc \
    --trainer --models="ResGatedGraphConv" --use_edge_attr \
    --data_path=benchmark_data --exp_path=EXP_full_3_1 \
    --clean # --logging