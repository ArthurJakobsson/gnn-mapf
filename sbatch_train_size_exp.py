import subprocess
import argparse
# multiprocessing libs
from multiprocessing import Pool
from itertools import repeat
from custom_utils.argparser_main import parse_arguments


def run_sbatch(my_input):
    dataset_size, args = my_input
    command = "python -m sbatch_master_process_runner"
    for var in vars(args):
        key, value = var, getattr(args, var)
        if key=="expName":
            value = args.expName
        if key=="dataset_size_list":
            continue
        command+= " --{}={}".format(key, value)
    command += f" --dataset_size={dataset_size}"
    subprocess.run(command.split(" "))

if __name__ == "__main__":
    flags = [
        "expnum",
        "mini_test",
        "numScensToCreate",
        "num_parallel",
        "data_folder",
        "k",
        "m",
        "lr",
        "batch_size",
        "relu_type",
        "expName",
        "numAgents",
        "extra_layers",
        "bd_pred",
        "which_setting",
        "percent_for_succ",
        "which_section",
        "iternum",
        "timeLimit",
        "num_scens",
        "dataset_size_list"
    ]

    args = parse_arguments(flags)
    dataset_sizes = [int(x) for x in args.dataset_size_list.split(",")] #[1,2,4,8,16,32,64,128]

    with Pool() as pool:
        results = pool.map(run_sbatch, zip(dataset_sizes, repeat(args)))
