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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import cv2
import re
import itertools

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

num_datapoints = {
    "1_no_bd": 6223,
    "4_no_bd": 22844,
    "1_CSFreeze": 6223,
    "2_CSFreeze": 12085,
    "4_CSFreeze": 22844,
    "8_CSFreeze": 44630,
    "16_CSFreeze": 88802,
    "32_CSFreeze": 177604,
    "64_CSFreeze": 350747,
    "128_CSFreeze": 712751,
    "1_agents": 6223,
    "2_agents": 12085,
    "4_agents": 22844,
    "8_agents": 44630,
    "16_agents": 88802,
    "32_agents": 177604,
    "64_agents": 350747,
    "128_agents": 712751,
    "1_big_bad_model": 6223,
    "4_big_bad_model": 22844,
    "1.2_eecbs": 22844,
    "1.5_eecbs": 22844,
    "2.0_eecbs": 22844
}

for key,val in num_datapoints.items():
    num_datapoints[key] = val*0.8 #train set is 80% of all data
    
# unheld_maps = ["Berlin_1_256", "empty_32_32", "maze_32_32_4", "random_64_64_20", "warehouse_20_40_10_2_1", "room_64_64_16"]
# held_maps = ["Paris_1_256", "empty_48_48", "maze_128_128_2", "random_64_64_10", "warehouse_10_20_10_2_1", "den312d"]


# unheld_maps = ["Berlin_1_256","Boston_0_256","empty_16_16","empty_32_32","ht_chantry","ht_mansion_n","lak303d","lt_gallowstemplar_n","maze_32_32_2","maze_32_32_4","ost003d","random_32_32_20","random_64_64_20","room_32_32_4","room_64_64_16","room_64_64_8","warehouse_10_20_10_2_2","warehouse_20_40_10_2_1","warehouse_20_40_10_2_2"]

held_maps = ["Paris_1_256", "den312d", "den520d","empty_48_48", "maze_128_128_2", "random_32_32_10", "random_64_64_10", "warehouse_10_20_10_2_1"]

unheld_maps = [
                "room_32_32_4",
                "room_64_64_16",
                "room_64_64_8"]
                # ["warehouse_10_20_10_2_1",
                #     "warehouse_10_20_10_2_2",
                #     "warehouse_20_40_10_2_1",
                #     "warehouse_20_40_10_2_2",
                #     "random_32_32_10",
                #     "random_32_32_20",
                #     "random_64_64_10",
                #     "random_64_64_20"
                #     ]
                # 
                # ["den312d",
                # "den520d",
                # "empty_16_16",
                # "empty_32_32",
                # "empty_48_48",
                # "maze_32_32_2",
                # "maze_32_32_4",
                # "maze_128_128_2"]
                # ["Berlin_1_256",
                # "Boston_0_256",
                # "Paris_1_256",
                # "ht_chantry",
                # "ht_mansion_n",
                # "lak303d",
                # "lt_gallowstemplar_n",
                # "ost003d"]#, \\
                # 
                # 

# maps = ["Berlin_1_256", "den312d"]
# maps = mapsToMaxNumAgents.keys()
maps_1 = unheld_maps
maps_2 = held_maps

# which_folders = ["benchmarking/big_run_results/benchmarking/1.2_eecbs_model_results", "benchmarking/big_run_results/benchmarking/1.5_eecbs_model_results", "benchmarking/big_run_results/benchmarking/2.0_eecbs_model_results"]
# which_folders = ["benchmarking/big_run_results/benchmarking/1_big_bad_model_results", "benchmarking/big_run_results/benchmarking/4_big_bad_model_results"]
# which_folders_cspibt = ["benchmarking/big_run_results/benchmarking/1_agents_results", "benchmarking/big_run_results/benchmarking/2_agents_results","benchmarking/big_run_results/benchmarking/4_agents_results","benchmarking/big_run_results/benchmarking/8_agents_results","benchmarking/big_run_results/benchmarking/16_agents_results","benchmarking/big_run_results/benchmarking/32_agents_results","benchmarking/big_run_results/benchmarking/64_agents_results","benchmarking/big_run_results/benchmarking/128_agents_results"]
# which_folders_csfreeze = ["benchmarking/big_run_results/benchmarking/1_CSFreeze_results", "benchmarking/big_run_results/benchmarking/2_CSFreeze_results","benchmarking/big_run_results/benchmarking/4_CSFreeze_results","benchmarking/big_run_results/benchmarking/8_CSFreeze_results","benchmarking/big_run_results/benchmarking/16_CSFreeze_results","benchmarking/big_run_results/benchmarking/32_CSFreeze_results","benchmarking/big_run_results/benchmarking/64_CSFreeze_results","benchmarking/big_run_results/benchmarking/128_CSFreeze_results"]
# which_folders = ["benchmarking/big_run_results/benchmarking/1_no_bd_results", "benchmarking/big_run_results/benchmarking/4_no_bd_results"]
# which_folders = ["benchmarking/big_run_results/benchmarking/1_big_bad_model_results", "benchmarking/big_run_results/benchmarking/4_big_bad_model_results", "benchmarking/big_run_results/benchmarking/1_agents_results", "benchmarking/big_run_results/benchmarking/4_agents_results"]
# which_folders = ["benchmarking/big_run_results/benchmarking/1_agents_results", "benchmarking/big_run_results/benchmarking/2_agents_results","benchmarking/big_run_results/benchmarking/4_agents_results","benchmarking/big_run_results/benchmarking/8_agents_results","benchmarking/big_run_results/benchmarking/16_agents_results","benchmarking/big_run_results/benchmarking/32_agents_results","benchmarking/big_run_results/benchmarking/64_agents_results","benchmarking/big_run_results/benchmarking/128_agents_results","benchmarking/big_run_results/benchmarking/1_CSFreeze_results", "benchmarking/big_run_results/benchmarking/2_CSFreeze_results","benchmarking/big_run_results/benchmarking/4_CSFreeze_results","benchmarking/big_run_results/benchmarking/8_CSFreeze_results","benchmarking/big_run_results/benchmarking/16_CSFreeze_results","benchmarking/big_run_results/benchmarking/32_CSFreeze_results","benchmarking/big_run_results/benchmarking/64_CSFreeze_results","benchmarking/big_run_results/benchmarking/128_CSFreeze_results"]

which_folders_cspibt_subset = ["benchmarking/big_run_results/benchmarking/1_agents_results","benchmarking/big_run_results/benchmarking/4_agents_results","benchmarking/big_run_results/benchmarking/16_agents_results","benchmarking/big_run_results/benchmarking/128_agents_results"]
which_folders_csfreeze_subset = ["benchmarking/big_run_results/benchmarking/1_CSFreeze_results","benchmarking/big_run_results/benchmarking/4_CSFreeze_results","benchmarking/big_run_results/benchmarking/16_CSFreeze_results","benchmarking/big_run_results/benchmarking/128_CSFreeze_results"]


which_folders = [i for j in zip(which_folders_cspibt_subset,which_folders_csfreeze_subset) for i in j]

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
                
                # print(f"Loaded {my_file} from {folder}")
            except Exception as e:
                print(f"Error loading {my_file}: {e}")
                
    pibt_folder = 'benchmarking/pibt_out/'
    pibt_file = os.path.join(pibt_folder, f"results_{which_map}_pibt.csv")
    pibt_df = pd.read_csv(pibt_file)
    data_frames.append(pibt_df)
    
    eph_folder = 'benchmarking/eph_results/'
    eph_file = os.path.join(eph_folder, f"{which_map}.csv")
    eph_df = pd.read_csv(eph_file)
    eph_df['Solution_Cost'] = None
    eph_df['Runtime'] = None
    data_frames.append(eph_df)

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        combined_df['Solution_Cost'] /= combined_df['Agent_Size']
        return combined_df
    else:
        return pd.DataFrame()  
    
def add_image_to_title(ax, mapfile, image_folder):
    # Load the image corresponding to the map
    img_path = os.path.join(image_folder, f"{mapfile.replace("_","-")}.png")
    if os.path.exists(img_path):
        image = plt.imread(img_path)
        image = cv2.resize(image, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        imagebox = OffsetImage(image, zoom=0.5)  # Adjust zoom for size
        ab = AnnotationBbox(imagebox, (0.9, 1.15), xycoords='axes fraction', frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    
# Assuming `result_data` is the DataFrame you're working with
def plot_success_rate(data, ax, mapname, info_type, last_row):
    # Filter the data for the specific programs
    filtered_data = data[data['Program'].isin(['LaCAM', 'PIBT', 'EECBS', 'GNNMAPF', 'EPH'])]
    
    marker_style = {
        'style':{
            'LaCAM': "1",
            'PIBT': "*",
            'EECBS': "^",
            'CS_Freeze': "x",
            'CS_PIBT': "o",
            'EPH': "s"
        },
        'color':{
            "EECBS": "blue",
            # "LaCAM": "red",
            "PIBT": "red",
            'EPH': "brown",
            "1": "darkcyan",
            "2": "pink",
            "4": "purple",
            "8": "olive",
            "16": "slateblue",
            "32": "gold",
            "64": "crimson",
            "128": "darksalmon"
        }
    }
    
    # Plot points for GNNMAPF and label with the folder it came from
    for folder in which_folders:
        folder_focus = folder.split('/')[-1]
        gnnmapf_data = filtered_data[filtered_data['source_folder'] == folder]
        gnnmapf_data = gnnmapf_data[gnnmapf_data['Program'] == 'GNNMAPF']
        cs_type = "CS_Freeze" if "CSFreeze" in folder_focus else "CS_PIBT"
        color = marker_style['color'][re.findall(r'\d+',folder_focus)[0]]
        linestyle = 'solid' if cs_type=="CS_PIBT" else 'dashed'
        ax.plot(gnnmapf_data['Agent_Size'], gnnmapf_data[info_type], label=f"GNNMAPF_{folder_focus}", marker=marker_style['style'][cs_type], color=color, linestyle=linestyle, alpha=0.5)
    
    for program in ['PIBT', 'EECBS', 'EPH']: # 'LaCAM'
        if (program == 'EPH') and (info_type != "Success_Rate"):
            continue
        program_data = filtered_data[filtered_data['Program'] == program]
        ax.plot(program_data['Agent_Size'], program_data[info_type], label=program, marker=marker_style['style'][program], color=marker_style['color'][program], alpha=0.5)
    
    # Set labels and title
    if last_row:
        ax.set_xlabel('# of Agents', fontsize=13)
    # ax.set_ylabel(info_type)
    # ax.set_title(f'{info_type} for {mapname}')
    return ax.get_legend_handles_labels()



def plot_all_maps_single_row(maps, which_folders, output_path, image_folder):
    # Create a 3x6 grid for subplots (3 rows for info types, 6 columns for maps)
    fig, axes = plt.subplots(3, len(maps), figsize=(15, 10))  # 3 rows (one per info type), 6 maps (columns)
    
    # Plot each info type in a separate row
    info_types = ["Success_Rate", "Runtime", "Solution_Cost"]
    # info_types = ["Solution_Cost"]
    firsttime = True
    
    for row, info_type in enumerate(info_types):
        for col, mapfile in enumerate(maps):
            # Load data for the map
            result_data = load_csv_data(mapfile, which_folders)
            # Plot the respective info type for this map on the respective axis
            if firsttime:
                handles, labels = plot_success_rate(result_data, axes[row, col], mapfile, info_type, row==2)
                firsttime=False
            else:
                plot_success_rate(result_data, axes[row, col], mapfile, info_type, row==2)
            
            add_image_to_title(axes[0, col], mapfile, image_folder)
        
        # Add big text on the left to indicate the info_type for the row
        display_info_type = info_type.replace("Solution_Cost", "Solution Cost per Agent").replace("_", " ")
        axes[row, 0].text(-0.2, 0.5, display_info_type, fontsize=13, va='center', rotation=90, transform=axes[row, 0].transAxes)

    for col, mapfile in enumerate(maps):
        if mapfile in held_maps:
            mapfile += "*"
        fig.text((col+0.1)/len(maps), 1.05, mapfile, fontsize=15, fontstyle='italic', ha='left')
        
    labels = [word.replace("GNNMAPF", "").replace("_", " ").replace("agents", "SSIL CS-PIBT").replace("CSFreeze", "SSIL CS-Naive").replace("results", "") for word in labels]
    
    # Add a legend for the last info type row
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
               fancybox=True, shadow=True, ncol=6, prop={'size': 15})
    
    plt.tight_layout(rect=[0, 0, 1, 1.1])
    plt.savefig(output_path, format="png", bbox_inches='tight')
    plt.close()

output_folder = "benchmarking/visualization_out"
image_folder = "benchmarking/mapf-png"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Generate the plot for all info types in a single image
output_path = f'{output_folder}/all_info_types_all_maps_grid.png'
plot_all_maps_single_row(maps_1, which_folders, output_path, image_folder)
    
    
# Main function to plot all maps in a single row per info type
# def plot_all_maps_single_row(maps, which_folders, output_path):
#     # Create a 3x6 grid for subplots (3 rows for info types, 6 columns for maps)
#     fig, axes = plt.subplots(3, len(maps), figsize=(25, 15))  # 3 rows (one per info type), 6 maps (columns)
    
#     # Plot each info type in a separate row
#     info_types = ["Success_Rate", "Runtime", "Solution_Cost"]
    
#     for row, info_type in enumerate(info_types):
#         for col, mapfile in enumerate(maps):
#             # Load data for the map
#             result_data = load_csv_data(mapfile, which_folders)

#             # Plot the respective info type for this map on the respective axis
#             handles, labels = plot_success_rate(result_data, axes[row, col], mapfile, info_type)

#     # Add a legend for the last info type row
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
#                fancybox=True, shadow=True, ncol=5)

#     # Adjust layout to prevent overlapping
#     plt.tight_layout()
#     plt.savefig(output_path, format="pdf", bbox_inches='tight')
#     plt.close()

# output_folder = "benchmarking/visualization_out"
# os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# # Generate the plot for all info types in a single image
# output_path = f'{output_folder}/all_info_types_all_maps_grid.pdf'
# plot_all_maps_single_row(maps, which_folders, output_path)




# def plot_two_rows_of_maps(maps_1, maps_2, which_folders, output_path, image_folder):
#     # Create a 6x6 grid for subplots (6 rows for info types, 6 columns for maps)
#     fig, axes = plt.subplots(6, len(maps_1), figsize=(30, 23))  # 6 rows (3 per set of maps), 6 maps (columns)
    
#     # Info types
#     info_types = ["Success_Rate", "Runtime", "Solution_Cost"]
    
#     # Plot for the first row of maps (maps_1)
#     firsttime = True
#     for row, info_type in enumerate(info_types):
#         for col, mapfile in enumerate(maps_1):
#             # Load data for the map
#             result_data = load_csv_data(mapfile, which_folders)
            
#             # Plot the respective info type for this map on the respective axis
#             if firsttime:
#                 handles, labels = plot_success_rate(result_data, axes[row, col], mapfile, info_type, row == 2)
#                 firsttime = False
#             else:
#                 plot_success_rate(result_data, axes[row, col], mapfile, info_type, row == 2)
            
#             # Add image next to the title
#             add_image_to_title(axes[0, col], mapfile, image_folder)

#         # Add text on the left to indicate the info_type for the row
#         display_info_type = info_type.replace("Solution_Cost", "Solution Cost per Agent").replace("_", " ")
#         axes[row, 0].text(-0.2, 0.5, display_info_type, fontsize=13, va='center', rotation=90, transform=axes[row, 0].transAxes)
    
#     # Plot for the second row of maps (maps_2)
#     for row, info_type in enumerate(info_types):
#         for col, mapfile in enumerate(maps_2):
#             # Load data for the map
#             result_data = load_csv_data(mapfile, which_folders)
            
#             # Plot the respective info type for this map on the respective axis (second set of rows)
#             plot_success_rate(result_data, axes[row + 3, col], mapfile, info_type, row == 2)
            
#             # Add image next to the title
#             add_image_to_title(axes[3, col], mapfile, image_folder)

#         # Add text on the left to indicate the info_type for the second row
#         display_info_type = info_type.replace("Solution_Cost", "Solution Cost per Agent").replace("_", " ")
#         axes[row + 3, 0].text(-0.2, 0.5, display_info_type, fontsize=13, va='center', rotation=90, transform=axes[row + 3, 0].transAxes)

#     # Add titles for the columns (map names)
#     for col, mapfile in enumerate(maps_1):
#         fig.text((col+0.15) / len(maps_1), 1.02, mapfile, fontsize=18, fontstyle='italic', ha='left')
    
#     for col, mapfile in enumerate(maps_2):
#         fig.text((col+0.15) / len(maps_2), 0.51, mapfile, fontsize=18, fontstyle='italic', ha='left')

#     labels = [word.replace("GNNMAPF", "").replace("_", " ").replace("agents", "SSIL CS-PIBT").replace("CSFreeze", "SSIL CS-Naive").replace("results", "") for word in labels]
    
#     # Add a legend for the last info type row
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
#                fancybox=True, shadow=True, ncol=6, prop={'size': 15})
    
#     plt.tight_layout(rect=[0, 0, 1, 1.05], h_pad=[-1.4, -1.4, -1.4, -0.5, -1.4, -1.4, -1.4])
#     plt.savefig(output_path, format="pdf", bbox_inches='tight')
#     plt.close()

# # Example of usage
# output_folder = "benchmarking/visualization_out"
# image_folder = "benchmarking/mapf-png"
# os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# # Generate the plot for all info types with two sets of maps
# output_path = f'{output_folder}/all_info_types_two_rows_of_maps_grid.pdf'
# plot_two_rows_of_maps(maps_1, maps_2, which_folders, output_path, image_folder)