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
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# maps = ["Berlin_1_256", "den312d"]
maps = mapsToMaxNumAgents.keys()

which_folders = ["benchmarking/big_run_results/benchmarking/1_CSFreeze_results", "benchmarking/big_run_results/benchmarking/2_CSFreeze_results", "benchmarking/big_run_results/benchmarking/4_CSFreeze_results","benchmarking/big_run_results/benchmarking/8_CSFreeze_results","benchmarking/big_run_results/benchmarking/16_CSFreeze_results","benchmarking/big_run_results/benchmarking/32_CSFreeze_results","benchmarking/big_run_results/benchmarking/64_CSFreeze_results","benchmarking/big_run_results/benchmarking/128_CSFreeze_results"]


def load_csv_data(which_map, which_folders):
    data_frames = []
    first_df_loaded = False
    for folder in which_folders:
        pattern = os.path.join(folder, f"results_{which_map}.csv")
        matching_files = glob.glob(pattern)
        for my_file in matching_files:
            try:
                df = pd.read_csv(my_file)
                df['source_folder'] = folder
                if not first_df_loaded:
                    first_df_loaded = True  # Mark first DataFrame as loaded
                    data_frames.append(df)
                else:
                    subset_df = df[df['Program']=='GNNMAPF']
                    data_frames.append(subset_df)
                
                print(f"Loaded {my_file} from {folder}")
            except Exception as e:
                print(f"Error loading {my_file}: {e}")
    

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()  

# Assuming `result_data` is the DataFrame you're working with
def plot_success_rate(data, output_path, mapname, info_type):
    # Filter the data for the specific programs
    filtered_data = data[data['Program'].isin(['LaCAM', 'PIBT', 'EECBS', 'GNNMAPF'])]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    marker_style = {
        'style':{
            'LaCAM': "1",
            'PIBT': "H",
            'EECBS': "^",
            'GNNMAPF': "*"
        },
        'color':{
            "EECBS": "blue",
            "GNNMAPF": "green",
            "LaCAM": "red",
            "PIBT": "orange"
        }}
    
    # Plot lines for LaCAM, PIBT, and EECBS
    for program in ['LaCAM', 'PIBT', 'EECBS']:
        program_data = filtered_data[filtered_data['Program'] == program]
        plt.plot(program_data['Agent_Size'], program_data[info_type], label=program, marker=marker_style['style'][program], color=marker_style['color'][program])
    
    # Plot points for GNNMAPF and label with the folder it came from
    for folder in which_folders:
        folder_focus = folder.split('/')[-1]
        gnnmapf_data = filtered_data[filtered_data['source_folder'] == folder]
        gnnmapf_data = gnnmapf_data[gnnmapf_data['Program'] == 'GNNMAPF']
        plt.plot(gnnmapf_data['Agent_Size'], gnnmapf_data[info_type], label=f"GNNMAPF_{folder_focus}",marker=marker_style['style']['GNNMAPF'])
    # plt.scatter(gnnmapf_data['Agent_Size'], gnnmapf_data['Success_Rate'], color='green', label='GNNMAPF')
    
    # for i, row in gnnmapf_data.iterrows():
    #     plt.text(row['Agent_Size'], row['Success_Rate'], row['source_folder'], fontsize=8, ha='right')

    # Set labels and title
    plt.xlabel('Agent Size')
    plt.ylabel(info_type)
    plt.title(f'{info_type} vs Agent Size for {mapfile}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Call the function to plot
for mapfile in maps:
    result_data = load_csv_data(mapfile, which_folders)
    output_folder = "benchmarking/visualization_out"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    output_file = os.path.join(output_folder,f"temp_{mapfile}.csv")

    # Save the DataFrame to CSV
    result_data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    # Check the loaded data
    for info_type in ["Success_Rate", "Runtime", "Solution_Cost"]:
        # print(result_data.head())
        output_path=f'benchmarking/visualization_out/{info_type}_plot_{mapfile}.png'
        plot_success_rate(result_data, output_path, mapfile, info_type)