#! /bin/bash
module load anaconda3
conda activate arthur_env
export MKL_SERVICE_FORCE_INTEL=1
python ../sbatch_train_size_exp.py --expnum=0 --mini_test=t --expName=EXP_153_full_128agents --data_folder=benchmark_unheld --num_parallel=64 --k=4 --m=5 --lr=0.001 --relu_type=leaky_relu --numAgents=config_example.json --which_setting=Arthur --extra_layers=agent_locations --bd_pred=t --which_section=begin --iternum=0 --percent_for_succ=0.4 --timeLimit=120 --num_scens=0 --dataset_size_list=1000,10000,100000
