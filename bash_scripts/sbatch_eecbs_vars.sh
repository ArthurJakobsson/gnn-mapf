#! /bin/bash
module load anaconda3
conda activate arthur_env
export MKL_SERVICE_FORCE_INTEL=1
python ../sbatch_eecbs_vars_exp.py --expnum=0 --mini_test=t --expName=EXP_vary_suboptimality --data_folder=benchmark_unheld --num_parallel=64 --k=4 --m=5 --lr=0.001 --relu_type=leaky_relu --numAgents=config_example.json --which_setting=Arthur --extra_layers=agent_locations --bd_pred=t --which_section=begin --iternum=0 --percent_for_succ=0.4 --timeLimit=120 --num_scens=4 --eecbs_suboptimalities=1.01,1.1,1.2,1.5,2
