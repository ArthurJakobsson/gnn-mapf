#! /bin/bash

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2

# DATA
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/mini_benchmark_data \
#     --temp_bd_path=$PROJECT/data/logs/EXP_Collect_BD_mini \
#     --exp_path=$PROJECT/data/logs/EXP_mini_priorities --iternum=0 \
#     --eecbs_batchrunner --constants_generator --dataloader \
#     --num_parallel_runs=128 \
#     --num_multi_inputs=3 \
#     --num_multi_outputs=2 \
#     --clean
# TRAIN
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/mini_benchmark_data \
#     --exp_path=$PROJECT/data/logs/EXP_mini_priorities --iternum=0 \
#     --trainer \
#     --models="ResGatedGraphConv" \
#     --use_edge_attr \
#     --num_multi_inputs=3 \
#     --num_multi_outputs=2 \
#     --clean

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2

# DATA
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/tiny_benchmark_data \
#     --temp_bd_path=$PROJECT/data/logs/EXP_Collect_BD_tiny \
#     --exp_path=$PROJECT/data/logs/EXP_tiny_priorities --iternum=0 \
#     --eecbs_batchrunner --constants_generator --dataloader \
#     --num_parallel_runs=128 \
#     --num_multi_inputs=3 \
#     --num_multi_outputs=2
# TRAIN
python sbatch_master_process_runner2.py --data_path=$PROJECT/data/tiny_benchmark_data \
    --exp_path=$PROJECT/data/logs/EXP_tiny_priorities --iternum=0 \
    --trainer \
    --models="ResGatedGraphConv" \
    --use_edge_attr \
    --num_multi_inputs=3 \
    --num_multi_outputs=2