import subprocess
import argparse
# multiprocessing libs
from multiprocessing import Pool
from itertools import repeat
from custom_utils.argparser_main import parse_arguments


def run_sbatch(my_input):
    suboptimality_lvl, args = my_input
    command = "python -m sbatch_master_process_runner"
    for var in vars(args):
        key, value = var, getattr(args, var)
        if key=="expName":
            value = args.expName+f"_{suboptimality_lvl}subopt"
        if key=="eecbs_suboptimalities":
            continue
        command+= " --{}={}".format(key, value)
    command += f" --suboptimality={suboptimality_lvl}"
    subprocess.run(command.split(" "))

if __name__ == "__main__":
    flags = [
        "expnum",
        'mini_test',
        'numScensToCreate',
        'num_parallel',
        'data_folder',
        'k',
        'm',
        'lr',
        'batch_size',
        'relu_type',
        'expName',
        'numAgents',
        'extra_layers',
        'bd_pred',
        'which_setting',
        'percent_for_succ',
        'which_section',
        'iternum',
        'timeLimit',
        'num_scens',
        'eecbs_suboptimalities'
    ]

    args = parse_arguments(flags)
    suboptimalities = [float(x) for x in args.eecbs_suboptimalities.split(",")] #[1,2,4,8,16,32,64,128]

    with Pool() as pool:
        results = pool.map(run_sbatch, zip(suboptimalities, repeat(args)))
