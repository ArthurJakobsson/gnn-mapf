#! /bin/bash
module load anaconda3
conda activate arthur_env
python master_process_runner.py 0 f f 20 200 --num_parallel=200
