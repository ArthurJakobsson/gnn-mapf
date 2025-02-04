#! /bin/bash

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/mini_benchmark_data \
#     --exp_path=$PROJECT/data/logs/EXP_mini_priorities --iternum=0 \
#     --eecbs_batchrunner --constants_generator --dataloader \
#     --num_parallel_runs=10
python sbatch_master_process_runner2.py --data_path=$PROJECT/data/benchmark_data_copy \
    --exp_path=$PROJECT/data/logs/EXP_full_priorities --iternum=0 \
    --eecbs_batchrunner --constants_generator --dataloader \
    --num_parallel_runs=10