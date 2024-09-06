import subprocess
import argparse
# multiprocessing libs
from multiprocessing import Pool
from itertools import repeat


def run_sbatch(num_scens, args):
    command = "python -m master_process_runner"
    for key, value in vars(args):
        if key=="expName":
            key = args.expName+f"_{num_scens}agents"
        command+= " --{}={}".format(key, value)
    subprocess.run(command.split(" "))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expnum", help="experiment number", type=int)
    parser.add_argument('--mini_test', type=str)
    # parser.add_argument('generate_initial', help="NOTE: We should NOT need to do this given constant_npzs/ folder", type=lambda x: bool(str2bool(x)))
    parser.add_argument('--numScensToCreate', type=int, help="number of scens to create per pymodel, see simulator2.py", default=20)
    parser.add_argument('--num_parallel', type=int)
    parser.add_argument('--data_folder', type=str, help="name of folder with data")
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--relu_type', type=str, default="relu")
    parser.add_argument('--expName', help="Name of the experiment, e.g. Test5", required=True)
    numAgentsHelp = "Number of agents per scen; [int1,int2,..] or `increment` for all agents up to the max or include .json for pulling from config file, see eecbs_batchrunner3.py "
    parser.add_argument('--numAgents', help=numAgentsHelp, type=str, required=True)
    extraLayersHelp = "Types of additional layers for training, comma separated. Options are: agent_locations, agent_goal, at_goal_grid"
    parser.add_argument('--extra_layers', help=extraLayersHelp, type=str, default=None)
    parser.add_argument('--bd_pred', type=str, default=None, help="bd_predictions added to NN, type anything if adding")
    parser.add_argument('--which_setting', help="[Arthur, Rishi, PSC]", required=True) # E.g. use --which_setting to determine using conda env or different aspects
    parser.add_argument('--percent_for_succ', help="percent decreased scen creation for success instances in simulation", type=float, required=True)
    parser.add_argument('--which_section', help="[begin, setup, train, simulate]", required=True)
    parser.add_argument('--iternum', type=int)
    parser.add_argument('--timeLimit', help="time limit for simulation cs-pibt (-1 for no limit)", type=int, required=True)
    parser.add_argument('--num_scens_list', help="number scens to include, for each map, in the train set", type=str, required=True)
    
    args = parser.parse_args()
    scen_nums = [int(x) for x in args.num_scens_list.split(",")] #[1,2,4,8,16,32,64,128]

    with Pool() as pool:
        results = pool.map(run_sbatch, zip(scen_nums, repeat(args)))
