
import argparse
from custom_utils.common_helper import str2bool

parser_dictionary = {
    "expnum": {"help": "experiment number", "type": int},
    "mini_test": {"type": lambda x: bool(str2bool(x))},
    "numScensToCreate": {"type": int, "help": "number of scens to create per pymodel, see simulator.py", "default": 20},
    "num_parallel": {"type": int, "help": "number of parallel processes to run", "default": 1},
    "data_folder": {"type": str, "help": "name of folder with data"},
    "k": {"type": int, "default": 4, "help": "radius of box around agent, box is (2kx1) x (2k+1)"}, 
    "m": {"type": int, "default": 5 , "help": "closest m agents to agent in question"},
    "lr": {"type": float, "default": 0.005, "help": "learning rate for training"},
    "batch_size": {"type": int, "default": 64, "help": "batch size for training"},
    "relu_type": {"type": str, "default": "relu"},
    "expName": {"help": "Name of the experiment, e.g. Test5", "required": True},
    "numAgents": {"help": "Number of agents per scen; [int1,int2,..] or `increment` for all agents up to the max or include .json for pulling from config file, see eecbs_batchrunner.py ", "type": str, "required": True},
    "extra_layers": {"help": "Types of additional layers for training, comma separated. Options are: agent_locations, agent_goal, at_goal_grid", "type": str, "default": None},
    "bd_pred": {"type": str, "default": None, "help": "bd_predictions added to NN, type anything if adding"},
    "which_setting": {"help": "[Arthur, Rishi, PSC]", "required": True}, # E.g. use --which_setting to determine using conda env or different aspects
    "percent_for_succ": {"help": "percent decreased scen creation for success instances in simulation", "type": float, "required": True},
    "which_section": {"help": "[begin, setup, train, simulate]", "required": True},
    "iternum": {"type": int},
    "timeLimit": {"help": "time limit for simulation cs-pibt (-1 for no limit)", "type": int, "required": True},
    "num_scens": {"help": "number scens to include, for each map, in the train set", "type": int, "required": True},
    "suboptimality": {"help": "eecbs suboptimality level", "type": float, "default": 2},
    "dataset_size": {"type": int, "default": -1},
    "eecbs_suboptimalities": {"help": "suboptimality levels for eecbs", "type": str, "default": "1,2,4,8,16,32,64,128"},
    "num_scens_list": {"help": "number of scens to include in the train set", "type": str, "default": "1,2,4,8,16,32,64,128"}
}

def parse_arguments(flags):
    parser = argparse.ArgumentParser()
    for key in flags:
        parser.add_argument(f"--{key}", **flags[key])
    return parser.parse_args()