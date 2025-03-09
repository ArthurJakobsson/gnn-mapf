#! /bin/bash

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2

# DATA
# python sbatch_master_process_runner2.py --machine=psc \
#     --collect_data \
#     --data_path=mini_benchmark_data --exp_path=EXP_full

# python sbatch_master_process_runner2.py --machine=psc \
#     --dataloader \
#     --data_path=mini_benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=0 --num_multi_outputs=1
# python sbatch_master_process_runner2.py --machine=psc \
#     --dataloader \
#     --data_path=mini_benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=3 --num_multi_outputs=1
# python sbatch_master_process_runner2.py --machine=psc \
#     --dataloader \
#     --data_path=mini_benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=3 --num_multi_outputs=2
# python sbatch_master_process_runner2.py --machine=psc \
#     --dataloader \
#     --data_path=mini_benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=0 --num_multi_outputs=2

# TRAIN
# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --model="ResGatedGraphConv" --use_edge_attr \
#     --data_path=benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=0 --num_multi_outputs=1 \
#     --logging    
# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --model="ResGatedGraphConv" --use_edge_attr \
#     --data_path=benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=3 --num_multi_outputs=1 \
#     --logging
# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --model="ResGatedGraphConv" \
#     --data_path=benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=0 --num_multi_outputs=1 \
#     --logging    
# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --model="ResGatedGraphConv" \
#     --data_path=benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=3 --num_multi_outputs=1 \
#     --logging
# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --model="ResGatedGraphConv" --use_edge_attr \
#     --data_path=benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=3 --num_multi_outputs=2 \
#     --logging    
# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --model="ResGatedGraphConv" \
#     --data_path=benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=3 --num_multi_outputs=2 \
#     --logging
# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --model="ResGatedGraphConv" --use_edge_attr \
#     --data_path=benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=0 --num_multi_outputs=2 \
#     --logging    
# python sbatch_master_process_runner2.py --machine=psc \
#     --trainer --model="ResGatedGraphConv" \
#     --data_path=benchmark_data --exp_path=EXP_full \
#     --num_multi_inputs=0 --num_multi_outputs=2 \
#     --loggings