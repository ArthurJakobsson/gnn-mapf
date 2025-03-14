#!/bin/bash

module load anaconda3/2022.10
conda activate $PROJECT/.conda/envs/gnn-mapf-dev2
export MKL_SERVICE_FORCE_INTEL=1

# python -m data_collection.maze_generator --data_path=$PROJECT/data/maze_benchmark_data/ \
#         --temp_bd_path=$PROJECT/data/logs/EXP_Generate_mazes/ \
#         --maze_config_csv=$PROJECT/data/mazes_test.csv \
#         --eecbs_path=./data_collection/eecbs/build_release5/eecbs --skip_octile_bfs

python -m data_collection.data_manipulator --bdIn=/ocean/projects/cis240049p/mliu13/data/logs/EXP_Generate_mazes///maze4_32_32_1/bd --goalsOutFile=/ocean/projects/cis240049p/mliu13/data/maze_benchmark_data//constant_npzs/maze4_32_32_1_goals.npz --bdOutFile=/ocean/projects/cis240049p/mliu13/data/maze_benchmark_data//constant_npzs/maze4_32_32_1_bds.npz --scenIn=/ocean/projects/cis240049p/mliu13/data/maze_benchmark_data//scens --mapIn=/ocean/projects/cis240049p/mliu13/data/maze_benchmark_data//maps --mapOutFile=/ocean/projects/cis240049p/mliu13/data/maze_benchmark_data//constant_npzs/all_maps.npz --num_parallel=1        

# python -m data_collection.eecbs_batchrunner5 --mapFolder=$PROJECT/data/mini_benchmark_data/maps \
#     --scenFolder=$PROJECT/data/mini_benchmark_data/scens \
#     --numAgents=50,100 \
#     --outputFolder=$PROJECT/data/logs/EXP_mini/iter0/eecbs_outputs \
#     --num_parallel_runs=1 \
#     "eecbs" \
#     --eecbsPath=./data_collection/eecbs/build_release5/eecbs \
#     --outputPathNpzFolder=$PROJECT/data/logs/EXP_mini/iter0/eecbs_npzs \
#     --firstIter=true --cutoffTime=5

# python -m data_collection.constants_generator --mapFolder=$PROJECT/data/mini_benchmark_data/maps \
#         --scenFolder=$PROJECT/data/mini_benchmark_data/scens \
#         --constantMapAndBDFolder=$PROJECT/data/mini_benchmark_data/constant_npzs \
#         --outputFolder=$PROJECT/data/logs/EXP_Collect_BD \
#         --num_parallel_runs=1 \
#         --deleteTextFiles=true \
#         "eecbs" \
#         --eecbsPath=./data_collection/eecbs/build_release5/eecbs \
#         --firstIter=true --cutoffTime=1

# python -m gnn.dataloader --mapNpzFile=$PROJECT/data/mini_benchmark_data/constant_npzs/all_maps.npz \
#       --bdNpzFolder=$PROJECT/data/mini_benchmark_data/constant_npzs \
#       --pathNpzFolder=$PROJECT/data/logs/EXP_mini/iter0/eecbs_npzs \
#       --processedFolder=$PROJECT/data/logs/EXP_mini/iter0/processed_0_1 \
#       --k=5 \
#       --m=3 \
#       --num_priority_copies=10 \
#       --num_multi_inputs=0 \
#       --num_multi_outputs=1 --bd_pred
# python -m gnn.dataloader --mapNpzFile=$PROJECT/data/mini_benchmark_data/constant_npzs/all_maps.npz \
#       --bdNpzFolder=$PROJECT/data/mini_benchmark_data/constant_npzs \
#       --pathNpzFolder=$PROJECT/data/logs/EXP_mini/iter0/eecbs_npzs \
#       --processedFolder=$PROJECT/data/logs/EXP_mini/iter0/processed_3_1 \
#       --k=5 \
#       --m=3 \
#       --num_priority_copies=10 \
#       --num_multi_inputs=3 \
#       --num_multi_outputs=1 --bd_pred
# python -m gnn.dataloader --mapNpzFile=$PROJECT/data/mini_benchmark_data/constant_npzs/all_maps.npz \
#       --bdNpzFolder=$PROJECT/data/mini_benchmark_data/constant_npzs \
#       --pathNpzFolder=$PROJECT/data/logs/EXP_mini/iter0/eecbs_npzs \
#       --processedFolder=$PROJECT/data/logs/EXP_mini/iter0/processed_0_2 \
#       --k=5 \
#       --m=3 \
#       --num_priority_copies=10 \
#       --num_multi_inputs=0 \
#       --num_multi_outputs=2 --bd_pred
# python -m gnn.dataloader --mapNpzFile=$PROJECT/data/mini_benchmark_data/constant_npzs/all_maps.npz \
#       --bdNpzFolder=$PROJECT/data/mini_benchmark_data/constant_npzs \
#       --pathNpzFolder=$PROJECT/data/logs/EXP_mini/iter0/eecbs_npzs \
#       --processedFolder=$PROJECT/data/logs/EXP_mini/iter0/processed_3_2 \
#       --k=5 \
#       --m=3 \
#       --num_priority_copies=10 \
#       --num_multi_inputs=3 \
#       --num_multi_outputs=2 --bd_pred

# python -m gnn.trainer --exp_folder=$PROJECT/data/logs/EXP_mini --experiment=exp0 --iternum=0 --num_cores=4 \
#   --processedFolders=$PROJECT/data/logs/EXP_mini/iter0/processed_0_1 \
#   --k=5 --m=3 --lr=0.01 \
#   --num_priority_copies=10 \
#   --num_multi_inputs=3 \
#   --num_multi_outputs=1 --bd_pred \
#   --gnn_name=ResGatedGraphConv \
#   --use_edge_attr
# python -m gnn.trainer --exp_folder=$PROJECT/data/logs/EXP_mini --experiment=exp0 --iternum=0 --num_cores=4 \
#   --processedFolders=$PROJECT/data/logs/EXP_mini/iter0/processed_3_1 \
#   --k=5 --m=3 --lr=0.01 \
#   --num_priority_copies=10 \
#   --num_multi_inputs=3 \
#   --num_multi_outputs=1 --bd_pred \
#   --gnn_name=ResGatedGraphConv \
#   --use_edge_attr

#   python -m gnn.simulator3 --mapNpzFile=$PROJECT/data/maze_benchmark_data/constant_npzs/all_maps.npz \
#         --mapName=maze4_32_32_1 --scenFile=$PROJECT/data/maze_benchmark_data/scens/maze4_32_32_1-random-1.scen \
#         --agentNum=4 --bdPath=$PROJECT/data/maze_benchmark_data/constant_npzs/\
#         --k=5 --m=3 \
#         --outputCSVFile=$PROJECT/data/logs/EXP_mini/tests/results.csv \
#         --outputPathsFile=$PROJECT/data/logs/EXP_mini/tests/encountered_scens/paths.npy \
#         --numScensToCreate=10 --outputScenPrefix=$PROJECT/data/logs/EXP_mini/iter0/encountered_scens/den520d/den520d-random-1.scen100 \
#         --maxSteps=400 --seed=0 --lacamLookahead=5 --timeLimit=100 --bd_pred \
#         --num_priority_copies=10 \
#         --useGPU=False --modelPath=$PROJECT/data/logs/EXP_mini/iter0/models_ResGatedGraphConv_0_1_p/max_test_acc.pt \
#         --num_multi_inputs=0 --num_multi_outputs=1 --shieldType=CS-PIBT 
# python -m gnn.simulator3 --mapNpzFile=$PROJECT/data/maze_benchmark_data/constant_npzs/all_maps.npz \
#         --mapName=maze1_16_16_1 --scenFile=$PROJECT/data/maze_benchmark_data/scens/maze1_16_16_1-random-1.scen \
#         --agentNum=10 --bdPath=$PROJECT/data/maze_benchmark_data/constant_npzs/\
#         --k=5 --m=3 \
#         --outputCSVFile=$PROJECT/data/logs/EXP_mini/tests/results.csv \
#         --outputPathsFile=$PROJECT/data/logs/EXP_mini/tests/encountered_scens/paths.npy \
#         --numScensToCreate=10 --outputScenPrefix=$PROJECT/data/logs/EXP_mini/iter0/encountered_scens/den520d/den520d-random-1.scen100 \
#         --maxSteps=400 --seed=0 --lacamLookahead=5 --timeLimit=100 --bd_pred \
#         --num_priority_copies=10 \
#         --useGPU=False --modelPath=$PROJECT/data/logs/EXP_mini/iter0/models_ResGatedGraphConv_3_1_p/max_test_acc.pt \
#         --num_multi_inputs=3 --num_multi_outputs=1 --shieldType=CS-PIBT 