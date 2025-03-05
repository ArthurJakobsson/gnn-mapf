#! /bin/bash

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2

# DATA
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
#     --temp_bd_path=$PROJECT/data/logs/EXP_Collect_BD_medium \
#     --exp_path=$PROJECT/data/logs/EXP_medium_priorities_1_1 --iternum=0 \
#     --eecbs_batchrunner --constants_generator --dataloader \
#     --num_parallel_runs=128 \
#     --num_multi_inputs=1 \
#     --num_multi_outputs=1 \
#     --clean
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
#     --temp_bd_path=$PROJECT/data/logs/EXP_Collect_BD_medium \
#     --exp_path=$PROJECT/data/logs/EXP_medium_priorities_3_1 --iternum=0 \
#     --eecbs_batchrunner --constants_generator --dataloader \
#     --num_parallel_runs=128 \
#     --num_multi_inputs=3 \
#     --num_multi_outputs=1 \
#     --clean
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
#     --temp_bd_path=$PROJECT/data/logs/EXP_Collect_BD_medium \
#     --exp_path=$PROJECT/data/logs/EXP_medium_priorities_3_2 --iternum=0 \
#     --eecbs_batchrunner --constants_generator --dataloader \
#     --num_parallel_runs=128 \
#     --num_multi_inputs=3 \
#     --num_multi_outputs=2 \
#     --clean
    
# TRAIN
python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
    --exp_path=$PROJECT/data/logs/EXP_medium_priorities_1_1 --iternum=0 \
    --trainer \
    --models="ResGatedGraphConv" \
    --use_edge_attr \
    --logging \
    --clean
python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
    --exp_path=$PROJECT/data/logs/EXP_medium_1_1 --iternum=0 \
    --trainer \
    --models="ResGatedGraphConv" \
    --logging \
    --clean

python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
    --exp_path=$PROJECT/data/logs/EXP_medium_priorities_3_1 --iternum=0 \
    --trainer \
    --models="ResGatedGraphConv" \
    --num_multi_inputs=3 \
    --use_edge_attr \
    --logging \
    --clean
python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
    --exp_path=$PROJECT/data/logs/EXP_medium_3_1 --iternum=0 \
    --trainer \
    --models="ResGatedGraphConv" \
    --num_multi_inputs=3 \
    --logging \
    --clean

# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
#     --exp_path=$PROJECT/data/logs/EXP_medium_priorities_3_2 --iternum=0 \
#     --trainer \
#     --models="ResGatedGraphConv" \
#     --num_multi_inputs=3 \
#     --num_multi_outputs=2 \
#     --use_edge_attr \
#     --logging
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
#     --exp_path=$PROJECT/data/logs/EXP_medium_priorities_3_2 --iternum=0 \
#     --trainer \
#     --models="ResGatedGraphConv" \
#     --num_multi_inputs=3 \
#     --num_multi_outputs=2 \
#     --logging

    # ResGatedGraphConv
    