import argparse
import subprocess
import pandas as pd
import os

def run_eecbs(scen, num_agents, args):
    pass

def run_lacam(scen, num_agents, args):
    command = "benchmarking/lacam/build/main"
    lacam_dict = {
        "m": f"{args.map_folder}/maps/{args.mapname}",
        "i": f"{args.data_folder}/scens/{scen}",
        "N": str(num_agents),
        "o": f"./build/{args.mapname}_{scen}_{num_agents}"
    }
    
    # run command with subprocess
    # TODO: load in the txt from "o" and parse it to get succ rate, solution_cost and runtime

def run_eph(scen, num_agents, args):
    pass

def run_gnn_mapf(scen, num_agents, args):
    pass

def run_single_instance(scen, num_agents, which_program, args):
    
    if which_program == "EECBS":
        success_rate, solution_cost, runtime = run_eecbs(scen, num_agents, args)
    elif which_program == "LaCAM":
        success_rate, solution_cost, runtime = run_lacam(scen, num_agents, args)
    elif which_program == "EPH":
        success_rate, solution_cost, runtime = run_eph(scen, num_agents, args)
    else: #ours
        success_rate, solution_cost, runtime = run_gnn_mapf(scen, num_agents, args)
    
    return success_rate, solution_cost, runtime


def get_scens(folder_path, map_name):
    # List to store files that match the criteria
    matching_files = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the filename contains the substring
        if map_name in filename:
            matching_files.append(filename)

    return matching_files


def main(args, mapname, agent_sizes):
    # Create a folder to save results if it doesn't exist
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    
    # DataFrame to hold results
    columns = ['Agent Size', 'Program', 'Success Rate', 'Solution Cost', 'Runtime']
    results_df = pd.DataFrame(columns=columns)
    
    scen_names = get_scens(args.mapname)
    
    # Iterate through each agent size and run the three programs
    for num_agent in agent_sizes:
        for program in ['EECBS', 'LaCAM', 'EPH', 'GNNMAPF']:
            for scen in scen_names:
                success_rate, solution_cost, runtime  = run_single_instance(scen, num_agent, program, args)
                
                # Append the results to the DataFrame
                results_df = results_df.append({
                    'Agent Size': num_agent,
                    'Program': program,
                    'Success Rate': success_rate,
                    'Solution Cost': solution_cost,
                    'Runtime': runtime
                }, ignore_index=True)
    
    # Save DataFrame to CSV
    csv_filename = os.path.join(results_folder, f'results_{mapname}.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f'Results saved to {csv_filename}')

if __name__ == '__main__':
    
    data_folder = "data_collection/data/benchmark_data"
    scen_folder = data_folder + "/scens"
    map_folder = data_folder + "/maps"
    
    parser = argparse.ArgumentParser(description='Run programs on different agent sizes and save results.')
    parser.add_argument('mapname', type=str, help='Name of the map to use.')
    parser.add_argument('agent_sizes', type=int, nargs='+', help='List of agent sizes to try.')
    parser.add_argument('data_folder', type=str, help="directory of data", default = data_folder)
    parser.add_argument('map_folder', type=str, help="directory of maps", default = map_folder)
    parser.add_argument('scen_folder', type=str, help="directory of scens", default = scen_folder)
    
    args = parser.parse_args()
    main(args, args.mapname, args.agent_sizes)