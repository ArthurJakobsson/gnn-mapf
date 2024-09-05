import numpy as np
from scipy.ndimage import label
# Optionally, visualize the map using matplotlib
import matplotlib.pyplot as plt
import pdb


def parse_map(mapfile):
    '''
    takes in a mapfile and returns a parsed np array
    '''
    with open(mapfile) as f:
        line = f.readline()  # "type octile"

        line = f.readline()  # "height 32"
        height = int(line.split(' ')[1])

        line = f.readline()  # width 32
        width = int(line.split(' ')[1])

        line = f.readline()  # "map\n"
        assert(line == "map\n")

        mapdata = np.array([list(line.rstrip()) for line in f])

    mapdata.reshape((width,height))
    mapdata[mapdata == '.'] = 0
    mapdata[mapdata == '@'] = 1
    mapdata[mapdata == 'T'] = 1
    mapdata = mapdata.astype(int)
    return mapdata

def parse_scene(scen_file):
    """Input: scenfile
    Output: start_locations, goal_locations
    """
    start_locations = []
    goal_locations = []

    with open(scen_file) as f:
        line = f.readline().strip()
        # if line[0] == 'v':  # Nathan's benchmark
        start_locations = list()
        goal_locations = list()
        # sep = "\t"
        for line in f:
            line = line.rstrip() #remove the new line
            # line = line.replace("\t", " ") # Most instances have tabs, but some have spaces
            # tokens = line.split(" ")
            tokens = line.split("\t") # Everything is tab separated
            assert(len(tokens) == 9) 
            # num_of_cols = int(tokens[2])
            # num_of_rows = int(tokens[3])
            tokens = tokens[4:]  # Skip the first four elements
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row,col)) # This is consistent with usage
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row,col)) # This is consistant with usage
    return np.array(start_locations, dtype=int), np.array(goal_locations, dtype=int)


def draw_grid_with_agents(map_grid, goals, start_locations, filename, cell_size=1):
    # Define the grid size
    grid_size = len(map_grid), len(map_grid[0])
    
    # Adjust the figure size based on the cell size
    fig, ax = plt.subplots(figsize=(grid_size[1] * cell_size, grid_size[0] * cell_size))
    
    # Draw the map
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            color = 'black' if map_grid[i][j] == 1 else 'white'
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)
    
    # Overlay goals (using stars)
    for idx, goal in enumerate(goals):
        ax.plot(goal[1] + 0.5, goal[0] + 0.5, marker='*', color=f'C{idx}', markersize=30)
    
    # Overlay start locations (using circles)
    for idx, start in enumerate(start_locations):
        ax.plot(start[1] + 0.5, start[0] + 0.5, marker='o', color=f'C{idx}', markersize=20)
    
    # Set limits and grid lines
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks(np.arange(0, grid_size[1], 1))
    ax.set_yticks(np.arange(0, grid_size[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    
    # Invert the y-axis to match the conventional grid layout
    ax.invert_yaxis()
    
    # Adjust the aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save the plot to a file
    plt.savefig(filename, bbox_inches='tight')
    
    # Close the plot to avoid display
    plt.close()
    
def my_special_function():
    mapname= "Berlin_1_256"
    scen ="Berlin_1_256-random-129.scen"
    my_map = parse_map(f"data_collection/data/benchmark_data/maps/{mapname}.map")
    start_loc, goal_loc = parse_scene(f"data_collection/data/benchmark_data/scens/{scen}")
    draw_grid_with_agents(my_map, goal_loc, start_loc, "new_map_generation/129.png")
    

import os
import re

def delete_files_in_folder(folder_path):
    # Get the list of all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file has the correct format
        match = re.search(r'-(\d+)\.scen$', filename)
        if match:
            # Extract the number from the filename
            number = int(match.group(1))
            # Check if the number is greater than 25
            if number > 25:
                # Construct the full file path
                file_path = os.path.join(folder_path, filename)
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Specify the folder path
# folder_path = 'data_collection/data/benchmark_data/scens'

# Run the function to delete files
# delete_files_in_folder(folder_path)