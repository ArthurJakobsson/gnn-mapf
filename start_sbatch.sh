#! /bin/bash
module load anaconda3
conda activate arthur_env
export MKL_SERVICE_FORCE_INTEL=1
python sbatch_master_process_runner.py --expnum=0 --mini_test=t --expName=EXP_test_sbatch  --data_folder=den312_benchmark --num_parallel=50 --k=4 --m=7 --lr=0.001 --relu_type=leaky_relu --numAgents=100,200,300,400 --which_setting=Arthur --extra_layers=agent_locations --bd_pred=t --which_section=begin --iternum=0
