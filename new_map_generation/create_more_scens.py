import numpy as np
from scipy.ndimage import label
# Optionally, visualize the map using matplotlib
import matplotlib.pyplot as plt
import pdb
import os
from collections import deque
from multiprocessing import Pool


mapsToMaxNumAgents = {
    "Berlin_1_256": 1000,
    "Boston_0_256": 1000,
    "Paris_1_256": 1000,
    "brc202d": 1000,
    "den312d": 1000, 
    "den520d": 1000,
    "dense_map_15_15_0":50,
    "dense_map_15_15_1":50,
    "corridor_30_30_0":50,
    "empty_8_8": 32,
    "empty_16_16": 128,
    "empty_32_32": 512,
    "empty_48_48": 1000,
    "ht_chantry": 1000,
    "ht_mansion_n": 1000,
    "lak303d": 1000,
    "lt_gallowstemplar_n": 1000,
    "maze_128_128_1": 1000,
    "maze_128_128_10": 1000,
    "maze_128_128_2": 1000,
    "maze_32_32_2": 333,
    "maze_32_32_4": 395,
    "orz900d": 1000,
    "ost003d": 1000,
    "random_32_32_10": 461,
    "random_32_32_10_custom_0": 400,
    "random_32_32_10_custom_1": 400,
    "random_32_32_20": 400,
    "random_64_64_10_custom_0": 1000,
    "random_64_64_10_custom_1": 1000,
    "random_64_64_10": 1000,
    "random_64_64_20": 1000,
    "room_32_32_4": 341,
    "room_64_64_16": 1000,
    "room_64_64_8": 1000,
    "w_woundedcoast": 1000,
    "warehouse_10_20_10_2_1": 1000,
    "warehouse_10_20_10_2_2": 1000,
    "warehouse_20_40_10_2_1": 1000,
    "warehouse_20_40_10_2_2": 1000,
}

def convert_to_map_format(array):
    # Convert the 2D array to a string format with '.' for 1s and '@' for 0s
    map_str = "\n".join("".join('.' if cell == 0 else '@' for cell in row) for row in array)
    
    # Add the required header information
    height, width = array.shape
    result = f"type octile\nheight {height}\nwidth {width}\nmap\n{map_str}"
    
    return result

def save_map_to_file(map_str, filename):
    with open(filename, 'w') as file:
        file.write(map_str)


def is_connected(map_grid, total_accessible_cells, start):
    rows, cols = len(map_grid), len(map_grid[0])
    
    # Directions for moving: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize visited grid
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    # BFS queue
    queue = deque([start])
    visited[start[0]][start[1]] = True
    
    # Count the number of accessible cells
    accessible_cells = 1
    
    
    while queue:
        current_row, current_col = queue.popleft()
        
        for dr, dc in directions:
            new_row, new_col = current_row + dr, current_col + dc
            
            # Check if the new cell is within bounds and not visited
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if not visited[new_row][new_col] and map_grid[new_row][new_col] == 0:
                    visited[new_row][new_col] = True
                    queue.append((new_row, new_col))
                    accessible_cells += 1
        # print(accessible_cells, total_accessible_cells)
        if accessible_cells > 0.4*total_accessible_cells:
            return True
    
    # If the number of accessible cells equals the total number of non-obstacle cells
    return accessible_cells > 0.4*total_accessible_cells

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

def createScenFile(locs, goal_locs, map_name, scenFilepath):
    """Input: 
        locs: (N,2)
        goal_locs: (N,2)
        map_name: name of the map
        scenFilepath: filepath to save scen
    """
    assert(locs.min() >= 0 and goal_locs.min() >= 0)

    ### Write scen file with the locs and goal_locs
    # Note we need to swap row:[0],col:[1] and save it as col,row
    with open(scenFilepath, 'w') as f:
        f.write(f"version {len(locs)}\n")
        for i in range(locs.shape[0]):
            f.write(f"0\t{map_name}\t{0}\t{0}\t{locs[i,1]}\t{locs[i,0]}\t{goal_locs[i,1]}\t{goal_locs[i,0]}\t0\n")
    print("Scen file created at: {}".format(scenFilepath))
    
def is_connected_task(args):
    r, c, my_map, open_cells = args
    if my_map[r, c] == 0:
        if not is_connected(my_map, open_cells, (r, c)):
            return r, c, True
    return r, c, False

def parallel_process_map(my_map, open_cells, num_processes=4):
    # Prepare the arguments for each task
    tasks = [(r, c, my_map, open_cells) for r in range(my_map.shape[0]) for c in range(my_map.shape[1])]

    # Create a multiprocessing pool
    with Pool(processes=num_processes) as pool:
        results = pool.map(is_connected_task, tasks)

    # Update the map based on the results
    for r, c, to_block in results:
        if to_block:
            my_map[r, c] = 1
    return my_map

if __name__ == "__main__":
    num_scens = 128
    map_folder = "data_collection/data/benchmark_data/maps"
    map_names = os.listdir(map_folder)

    # Parameters
    size = 15
    obstacle_prob = 0.2
    

    for mapname in map_names:  
        if "_custom_" not in mapname:
            continue  
        map_path = os.path.join(map_folder, mapname)
        my_map = parse_map(map_path)
        open_cells = np.count_nonzero(my_map==0)
        my_map = parallel_process_map(my_map, open_cells, 64)

        for j in range(26, num_scens+26):
            empty_space = np.where(my_map == 0) # return list of (r,c) locations
            num_agents = mapsToMaxNumAgents[mapname[:-4]]
            assert(len(empty_space[0]) >= num_agents)
            start_locs = np.random.choice(np.arange(len(empty_space[0])), num_agents, replace=False)
            start_rows, start_cols = empty_space[0][start_locs], empty_space[1][start_locs]
            start_locs = np.vstack([start_rows, start_cols]).T
            assert(np.all(my_map[start_rows, start_cols] == 0))
            
            goal_locs = np.random.choice(np.arange(len(empty_space[0])), num_agents, replace=False)
            goal_rows, goal_cols = empty_space[0][goal_locs], empty_space[1][goal_locs]
            goal_locs = np.vstack([goal_rows, goal_cols]).T
            assert(np.all(my_map[goal_rows, goal_cols] == 0))

            createScenFile(start_locs,goal_locs, mapname, f"new_map_generation/scens/{mapname[:-4]}-random-{j}.scen")
