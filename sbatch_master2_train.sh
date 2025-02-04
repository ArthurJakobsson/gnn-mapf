#! /bin/bash

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2
# python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
#     --exp_path=$PROJECT/data/logs/EXP_medium_priorities --iternum=0 \
#     --trainer \
#     --models="ResGatedGraphConv" \
#     --logging \
#     --use_edge_attr
python sbatch_master_process_runner2.py --data_path=$PROJECT/data/medium_benchmark_data \
    --exp_path=$PROJECT/data/logs/EXP_medium_priorities --iternum=0 \
    --trainer \
    --models="SAGEConv" \
    --logging \
    --clean
