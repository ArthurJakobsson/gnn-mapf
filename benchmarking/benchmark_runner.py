import argparse
import subprocess
import pandas as pd
import os
import tqdm

def run_eecbs(scen, num_agents, args):
    pass


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
            result["runtime"] = float(line.split('=')[1])
        elif line.startswith("solved"):
            result["success"] = bool(int(line.split('=')[1]))
        elif line.startswith("soc"):
            result["soc_cost"] = int(line.split('=')[1])
    return result

def run_lacam(scen, num_agents, args):
    command = "benchmarking/lacam/build/main"
    lacam_dict = {
        "m": f"{args.map_folder}/{args.mapname}.map",
        "i": f"{args.scen_folder}/{scen}",
        "N": str(num_agents),
        "o": f"benchmarking/lacam/build/result_txts/{args.mapname}_{scen}_{num_agents}.txt",
        "v": 0
    }
    
    for aKey in lacam_dict:
        command += f" -{aKey} {lacam_dict[aKey]}"
    
    subprocess.run(command, check=True, shell=True)
    
    result = parse_lacam_output(lacam_dict['o'])
    return result["success"], result["soc_cost"], result["runtime"]
    
    # run command with subprocess
    # TODO: load in the txt from "o" and parse it to get succ rate, solution_cost and runtime

def run_eph(scen, num_agents, args):
    pass

def run_gnn_mapf(scen, num_agents, args):
    pass

def run_single_instance(scen, num_agents, which_program, args):
    
    if which_program == "EECBS":
        success, solution_cost, runtime = run_eecbs(scen, num_agents, args)
    elif which_program == "LaCAM":
        success, solution_cost, runtime = run_lacam(scen, num_agents, args)
    elif which_program == "EPH":
        success, solution_cost, runtime = run_eph(scen, num_agents, args)
    else: #ours
        success, solution_cost, runtime = run_gnn_mapf(scen, num_agents, args)
    
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


def main(args, mapname, numAgents):
    # Create a folder to save results if it doesn't exist
    results_folder = "benchmarking/results"
    os.makedirs(results_folder, exist_ok=True)
    
    # DataFrame to hold results
    columns = ['Agent Size', 'Program', 'Success Rate', 'Solution Cost', 'Runtime']
    results_df = pd.DataFrame(columns=columns)
    
    scen_names = get_scens(args.scen_folder, args.mapname)
    
    # Iterate through each agent size and run the three programs
    for num_agent in numAgents:
        for program in ['LaCAM']: #['EECBS', 'LaCAM', 'EPH', 'GNNMAPF']:
            num_successes = 0
            total_runtime = 0
            total_solution_cost = 0
            num_scens = len(scen_names)
            for scen in tqdm(scen_names):
                success, solution_cost, runtime  = run_single_instance(scen, num_agent, program, args)
                num_successes += success
                total_solution_cost += solution_cost
                total_runtime += runtime
            success_rate = num_successes/num_scens
            runtime = total_runtime/num_successes
            solution_cost = total_solution_cost/num_successes #TODO: check that should be dividing by runtime (soc is 0 if failed i think)
            
            new_row={
                'Agent Size': num_agent,
                'Program': program,
                'Success Rate': success_rate,
                'Solution Cost': solution_cost,
                'Runtime': runtime
            }
            # Append the results to the DataFrame
            results_df =  pd.concat([results_df,pd.DataFrame([new_row])], ignore_index=True)
    
    # Save DataFrame to CSV
    csv_filename = os.path.join(results_folder, f'results_{mapname}.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f'Results saved to {csv_filename}')

if __name__ == '__main__':
    
    data_folder = "data_collection/data/benchmark_data"
    scen_folder = data_folder + "/scens"
    map_folder = data_folder + "/maps"
    
    parser = argparse.ArgumentParser(description='Run programs on different agent sizes and save results.')
    parser.add_argument('--mapname', type=str, help='Name of the map to use (without .map).')
    parser.add_argument('--numAgents', type=str, help='List of agent sizes to try.')
    parser.add_argument('--data_folder', type=str, help="directory of data", required=False, default = data_folder)
    parser.add_argument('--map_folder', type=str, help="directory of maps", required=False, default = map_folder)
    parser.add_argument('--scen_folder', type=str, help="directory of scens", required=False, default = scen_folder)
    
    args = parser.parse_args()
    args.numAgents = [int(x) for x in args.numAgents.split(",")]
    main(args, args.mapname, args.numAgents)