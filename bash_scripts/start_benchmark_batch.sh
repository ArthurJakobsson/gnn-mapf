#! /bin/bash
module load anaconda3
conda activate arthur_env
export MKL_SERVICE_FORCE_INTEL=1
python -m ../sbatch_simulator
