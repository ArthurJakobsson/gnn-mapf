import argparse
import subprocess
import pandas as pd
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import gc
from multiprocessing import Pool
import pdb
from itertools import repeat
import time
import torch_geometric.inspector


mapsToMaxNumAgents = {
    "Berlin_1_256": 1000,
    "Boston_0_256": 1000,
    "Paris_1_256": 1000,
    # "brc202d": 1000,
    "den312d": 1000, 
    "den520d": 1000,
    # "dense_map_15_15_0":50,
    # "dense_map_15_15_1":50,
    # "corridor_30_30_0":50,
    #"empty_8_8": 32,
    "empty_16_16": 128,
    "empty_32_32": 512,
    "empty_48_48": 1000,
    "ht_chantry": 1000,
    "ht_mansion_n": 1000,
    "lak303d": 1000,
    "lt_gallowstemplar_n": 1000,
    # "maze_128_128_1": 1000,
    # "maze_128_128_10": 1000,
    "maze_128_128_2": 1000,
    "maze_32_32_2": 333,
    "maze_32_32_4": 395,
    # "orz900d": 1000,
    "ost003d": 1000,
    "random_32_32_10": 461,
    "random_32_32_20": 409,
    "random_64_64_10": 1000,
    "random_64_64_20": 1000,
    "room_32_32_4": 341,
    "room_64_64_16": 1000,
    "room_64_64_8": 1000,
    #"w_woundedcoast": 1000,
    "warehouse_10_20_10_2_1": 1000,
    "warehouse_10_20_10_2_2": 1000,
    "warehouse_20_40_10_2_1": 1000,
    "warehouse_20_40_10_2_2": 1000,
}

def parse_lacam_output(filename):
    f = open(filename, "r")
    content = f.read()
    lines = content.strip().split('\n')
    result = {
        "runtime": None,
        "success": None,
        "soc_cost": None
    }
    for line in lines:
        if line.startswith("comp_time"):
            result["runtime"] = float(line.split('=')[1])/1000 # ms -> sec
            break #this is the last one that will show up
        elif line.startswith("solved"):
            result["success"] = bool(int(line.split('=')[1]))
        elif line.startswith("soc"):
            result["soc_cost"] = int(line.split('=')[1])
    return result

def run_lacam(scen,  mapname, num_agents, args):
    command = "benchmarking/lacam/build/main"
    lacam_dict = {
        "m": f"{args.map_folder}/{mapname}.map",
        "i": f"{args.scen_folder}/{scen}",
        "N": str(num_agents),
        "o": f"benchmarking/lacam/build/result_txts/{mapname}_{scen}_{num_agents}.txt",
        "v": 0,
        "t": 10
    }
    
    for aKey in lacam_dict:
        command += f" -{aKey} {lacam_dict[aKey]}"
    
    if not os.path.exists(f"benchmarking/lacam/build/result_txts/{mapname}_{scen}_{num_agents}.txt"):
        subprocess.run(command, check=True, shell=True)
        time.sleep(1)
    
    result = parse_lacam_output(lacam_dict['o'])
    return result["success"], result["soc_cost"], result["runtime"]

def fetch_eecbs(mapname, num_agent, args):
    eecbs_source = 'benchmarking/eecbs_less_old/eecbs_outputs/'
    eecbs_map_folder = eecbs_source + mapname + '/csvs/combined.csv'
    df = pd.read_csv(eecbs_map_folder)
    filtered_df = df[df['agentNum'] == num_agent]
    successes = filtered_df[filtered_df['solution cost'] > 0]
    successes = successes[successes['runtime'] < args.eecbs_cutoff]
    success_rate = len(successes.index)/len(filtered_df.index) if len(filtered_df.index)!=0 else 0
    runtime  = successes['runtime'].mean()
    solution_cost = successes['solution cost'].mean()
    return success_rate, solution_cost, runtime



def parse_eph(mapname, num_agent, eph_results):
    df = eph_results
    filtered_df = df[df['agentNum'] == num_agent]
    filtered_df = filtered_df[filtered_df['mapName'] == mapname]
    # successes = filtered_df[filtered_df['success']==True]
    
    success_rate = filtered_df['success']
    runtime  = filtered_df['runtime']
    solution_cost = filtered_df['total_cost_true']
    return success_rate, solution_cost, runtime 


def parse_pymodel_output(pymodel_output_folder, map_name, num_agents):
    print(pymodel_output_folder)
    df = pd.read_csv(f"{pymodel_output_folder}/{map_name}/csvs/combined.csv")
    filtered_df = df[df['agentNum'] == num_agents]
    successes = filtered_df[filtered_df['success']]
    success_rate = len(successes.index)/len(filtered_df.index) if len(filtered_df.index)!=0 else 0
    runtime  = successes['runtime'].mean()
    solution_cost = successes['total_cost_true'].mean()
    # 'create_nn_data', 'forward_pass', 'cs-time'
    create_nn_data = None
    forward_pass = None
    cs_time = None

    if 'create_nn_data' in successes.columns and 'forward_pass' in successes.columns and 'cs-time' in successes.columns
        create_nn_data = successes['create_nn_data'].mean()
        forward_pass = successes['forward_pass'].mean()
        cs_time = successes['cs-time'].mean()

    return success_rate, solution_cost, runtime, create_nn_data, forward_pass, cs_time

def run_gnn_mapf(mapname,num_agents, args):
    constantMapAndBDFolder = "data_collection/data/benchmark_original/constant_npzs"
    # mini data folder used for this experiment
    source_maps_scens = args.data_folder
    # the experiment folder
    # the iter we want to simulate with
    os.makedirs(args.pymodel_out, exist_ok=True)
    pymodel_output_folder = args.pymodel_out#"benchmarking/pymodel_results_test/"
    model_path = args.model_path
    # model_path = f"data_collection/data/logs/{args.expname}/iter{args.iternum}/models/max_test_acc.pt"
    
    command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
                f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                f"--numAgents={args.numAgents}",
                f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                f"--outputFolder={pymodel_output_folder}", 
                f"--num_parallel_runs={min(64, args.num_parallel)}",
                # f"--chosen_map={mapname}",
                "\"pymodel\"",
                f"--modelPath={model_path}",
                "--useGPU=False",
                "--k=4",
                "--m=5",
                "--maxSteps=3x",
                f"--shieldType={args.shieldType}",
                f"--timeLimit={args.simulator_cutoff}",
                # "--lacamLookahead=50",
                f"--numScensToCreate=0",
                f"--percentSuccessGenerationReduction=0"])
    if args.extra_layers is not None:
        command += f" --extra_layers={args.extra_layers}"
    if args.bd_pred is not None:
        command += f" --bd_pred={args.bd_pred}"       
    if args.conda_env is not None:
        command += f" --condaEnv={args.conda_env}"
    print(command)
    subprocess.run(command, shell=True, check=True)
    
    command = " ".join(["python", "-m", "data_collection.eecbs_batchrunner3", 
                f"--mapFolder={source_maps_scens}/maps",  f"--scenFolder={source_maps_scens}/scens",
                f"--numAgents={args.numAgents}",
                f"--constantMapAndBDFolder={constantMapAndBDFolder}",
                f"--outputFolder={pymodel_output_folder}", 
                f"--num_parallel_runs={args.num_parallel}",
                "\"clean\" --keepNpys=false"])
    # subprocess.run(command, shell=True, check=True)
    
    return parse_pymodel_output(pymodel_output_folder, mapname, num_agents)

def run_single_instance(scen, mapname, num_agents, which_program, args):
    
    # if which_program == "EECBS":
    #     success, solution_cost, runtime = run_eecbs(scen, num_agents, args)
    if which_program == "LaCAM":
        success, solution_cost, runtime = run_lacam(scen, mapname, num_agents, args)
    elif which_program == "EPH":
        success, solution_cost, runtime = parse_eph(scen, num_agents, args)
    
    return success, solution_cost, runtime


def get_scens(folder_path, map_name):
    # List to store files that match the criteria
    matching_files = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the filename contains the substring
        if map_name in filename:
            matching_files.append(filename)

    return matching_files

def get_num_agents(args, mapname):
    if args.numAgents == "increment":
        increment = min(100,  mapsToMaxNumAgents[mapname])
        maximumAgents = mapsToMaxNumAgents[mapname] + 1
        num_agents_list = list(range(increment, maximumAgents, increment))
    else:
        num_agents_list = [int(x) for x in args.numAgents.split(",")]
    return num_agents_list

def run_instance(info):
    scen, mapname, num_agent, program, args = info
    return run_single_instance(scen, mapname, num_agent, program, args)

def run_eph(args):
    # mapnames = str(list(mapsToMaxNumAgents.keys()))
    # print(mapnames)
    mapnames = "all" 
    agentNums = "increment"
    command = " ".join(["python", "-m", "eph_mapf.simulator_eph", 
                f"--mapname={mapnames}", f"--agentNums={agentNums}",
                "--debug=True", 
                "--modelPath=nothing", 
                "--useGPU=t",
                f"--timeLimit=500",#{args.simulator_cutoff}",
                "--outputCSVFile=./benchmarking/eph_results.csv"  
                ])
    print(command)
    subprocess.run(command, shell=True, check=True)


def main(args, mapname):
    print(mapname)
    # Create a folder to save results if it doesn't exist
    results_folder = f"benchmarking/{args.pymodel_out}"
    os.makedirs(results_folder, exist_ok=True)
    
    # DataFrame to hold results
    columns = ['Agent_Size', 'Program', 'Success_Rate', 'Solution_Cost', 'Runtime', 'Create_NN_Data', 'Forward_Pass', 'CS_Time']
    results_df = pd.DataFrame(columns=columns)
    
    scen_names = get_scens(args.scen_folder, mapname)
    num_agents_list = get_num_agents(args, mapname)
    # run_eph(args)
    # eph_results = pd.read_csv("./benchmarking/eph_results.csv")
        
    # Iterate through each agent size and run the three programs
    for num_agent in num_agents_list:
        for program in ['LaCAM', 'EECBS', 'GNNMAPF']: # ["GNNMAPF"]:# #['EECBS', 'LaCAM', 'EPH', 'GNNMAPF']:
            create_nn_data, forward_pass, cs_time = None, None, None # in case we are not running GNNMAPF
            if program=='EECBS':
                success_rate, solution_cost, runtime = fetch_eecbs(mapname, num_agent, args)
            elif program=='GNNMAPF':
                success_rate, solution_cost, runtime, create_nn_data, forward_pass, cs_time = run_gnn_mapf(mapname, num_agent, args)
            # elif program=="EPH":
            #     success_rate, solution_cost, runtime = parse_eph(mapname, num_agent, eph_results)
            else:
                assert(program=='LaCAM')
                
                num_successes = 0
                total_runtime = 0
                total_solution_cost = 0
                num_scens = len(scen_names)
                for scen in tqdm(scen_names):
                    success, solution_cost, runtime  = run_single_instance(scen, mapname, num_agent, program, args)
                    num_successes += success
                    if success:
                        total_solution_cost += solution_cost
                        total_runtime += runtime
                success_rate = num_successes / num_scens
                if num_successes > 0:
                    runtime = total_runtime / num_successes
                    solution_cost = total_solution_cost / num_successes
                else:
                    runtime = None
                    solution_cost = None
            # create_nn_data', 'forward_pass', 'cs-time'
            new_row={
                'Map_Name': mapname,
                'Agent_Size': num_agent,
                'Program': program,
                'Success_Rate': success_rate,
                'Solution_Cost': solution_cost,
                'Runtime': runtime,
                'Create_NN_Data': create_nn_data,
                'Forward_Pass': forward_pass,
                'CS_Time': cs_time
            }
            # Append the results to the DataFrame
            results_df =  pd.concat([results_df,pd.DataFrame([new_row])], ignore_index=True)
    
    # Save DataFrame to CSV
    csv_filename = os.path.join(results_folder, f'results_{mapname}.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f'Results saved to {csv_filename}')


def plot_all_maps_grid(mapnames, dfs, args, num_rows=5, num_cols=6):
    model_colors = {
        "EECBS": "blue",
        "GNNMAPF": "green",
        "LaCAM": "red"
    }
    held_maps = ["den312d","den520d", "empty_48_48", "maze_128_128_2", "Paris_1_256", "random_32_32_10", "random_64_64_10","warehouse_10_20_10_2_1"]
    
    models = set(dfs[0]['Program'].tolist())  # Assuming all DataFrames have the same models
    for column_name in ["Success_Rate", "Solution_Cost", "Runtime"]:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 15))  # Adjust figsize as needed
        # plt.subplots_adjust(left=0.7, bottom=0.7, right=0.8, top=0.8, wspace=0.8, hspace=0.8)
        fig.tight_layout(pad=3)
        axes = axes.flatten()  # Flatten to make indexing easier
        
        for i, (mapname, df) in enumerate(zip(mapnames, dfs)):
            ax = axes[i]
            if mapname in held_maps:
                ax.set_facecolor((1, 0.75, 0.75, 0.5))
            numAgents = get_num_agents(args,mapname)
            for model in models:
                plot_topic = df[df['Program'] == model][column_name].tolist()
                color = model_colors.get(model, 'black')
                ax.plot(numAgents, plot_topic, label=model, marker="*", color=color)
            
            ax.set_title(mapname)
            ax.set_ylabel(column_name)
            ax.set_xlabel("Number_Agents")
            ax.legend()
            # ax._in_layout()

        # Hide any unused subplots
        for j in range(i+1, num_rows*num_cols):
            fig.delaxes(axes[j])

        if not os.path.exists(f"benchmarking/{args.pymodel_out}/{column_name}"):
            os.makedirs(f"benchmarking/{args.pymodel_out}/{column_name}")
        
        plt.savefig(f"benchmarking/{args.pymodel_out}/{column_name}/all_maps_grid.pdf", format="pdf")
        plt.close(fig)  # Close the figure to free up memory
        gc.collect()
    

if __name__ == '__main__':
    
    data_folder = "data_collection/data/benchmark_data"
    scen_folder = data_folder + "/scens"
    map_folder = data_folder + "/maps"
    
    parser = argparse.ArgumentParser(description='Run programs on different agent sizes and save results.')
    parser.add_argument('--mapname', type=str, help='Name of the map to use (without .map).')
    parser.add_argument('--numAgents', type=str, help='List of agent sizes to try.')
    parser.add_argument('--data_folder', type=str, help="directory of data (prefix to map_folder and scen_folder)", required=False, default = data_folder)
    parser.add_argument('--map_folder', type=str, help="directory of maps", required=False, default = map_folder)
    parser.add_argument('--scen_folder', type=str, help="directory of scens", required=False, default = scen_folder)
    parser.add_argument('--extra_layers', type=str, help="extra_layers used in the model", required=True)
    parser.add_argument('--conda_env', type=str, help="conda env name", default=None)
    # parser.add_argument('--expname', type=str, help="name of model to run", required=True)
    parser.add_argument('--model_path', type=str, help="model path", required=True)
    parser.add_argument('--shieldType', type=str, default='CS-PIBT', choices=['CS-PIBT', 'CS-Freeze', 'LaCAM'])
    # parser.add_argument('--iternum', type=str, help="iteration of model to run", required=True)
    parser.add_argument('--num_parallel', type=int, help="number of parallel runs to do", default=20)
    parser.add_argument('--bd_pred', type=str, default=None, help="bd_predictions added to NN, type anything if adding")
    parser.add_argument('--eecbs_cutoff', type=int, required=True, help="num seconds for eecbs cutoff (<720)")
    parser.add_argument('--simulator_cutoff', type=int, required=True, help="num seconds for simulator cutoff")
    parser.add_argument('--pymodel_out', type=str, required=True, help="num seconds for simulator cutoff")


    args = parser.parse_args()
    
    if args.mapname=="all":
        all_maps = mapsToMaxNumAgents.keys()
    else:
        all_maps = args.mapname.split(",")
    
    run_eph(args)

    # get aggregate statistics for all maps
    results_df = []
    for mapname in all_maps:
        main(args, mapname)
        results_df.append(pd.read_csv(f"benchmarking/{args.pymodel_out}/results_{mapname}.csv"))
    plot_all_maps_grid(all_maps, results_df, args)
        
    
    # make central csv with results for all maps
    # final_df = None
    # for mapname in all_maps:
    #     results_df = pd.read_csv(f"benchmarking/results/results_{mapname}.csv")
    #     if not final_df:
    #         final_df = results_df
    #     else:
    #         final_df = pd.concat([final_df, results_df])
    
    # final_df.to_csv("benchmarking/results/all_maps.csv")
