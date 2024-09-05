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

def generate_connected_map(size, obstacle_prob):
    # Step 1: Generate a random map with obstacles
    # map = np.random.choice([0, 1], size=(size, size), p=[obstacle_prob, 1 - obstacle_prob])
    
    # # Step 2: Ensure the start and end points are free
    # map[0, 0] = 1
    # map[-1, -1] = 1
    
    # Step 3: Ensure full connectivity using a flood fill algorithm
    while True:
        map = np.random.choice([0, 1], size=(size, size), p=[obstacle_prob, 1 - obstacle_prob])

        labeled_map, num_features = label(map)
        
        # If the map is fully connected, num_features should be 1
        if num_features == 1:
            break
        
        # If not fully connected, remove an obstacle from the smallest disconnected region
        # sizes = np.bincount(labeled_map.ravel())
        # min_label = np.argmin(sizes[1:]) + 1  # find the smallest non-zero region
        # map[labeled_map == min_label] = 1
    
    map = 1 - map # invert obstacles and empty space
    return map


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


if __name__ == "__main__":
    num_maps = 2
    num_scens = 153
    num_agents = 500

    # Parameters
    size = 32
    obstacle_prob = 0.1

    for i in range(num_maps):

        my_map = generate_connected_map(size, obstacle_prob)
        print(my_map)
        map_str = convert_to_map_format(my_map)
        map_name = f"random_32_32_10_custom_{i}"
        save_map_to_file(map_str, "new_map_generation/maps/"+map_name+".map")
        # my_map = parse_map(f"new_map_generation/maps/random_64_64_10_{i}.map")
        # map_name=f"random_64_64_10_custom_{i}"

        for j in range(num_scens):

            empty_space = np.where(my_map == 0) # return list of (r,c) locations
            
            assert(len(empty_space[0]) >= num_agents)
            start_locs = np.random.choice(np.arange(len(empty_space[0])), num_agents, replace=False)
            start_rows, start_cols = empty_space[0][start_locs], empty_space[1][start_locs]
            start_locs = np.vstack([start_rows, start_cols]).T
            assert(np.all(my_map[start_rows, start_cols] == 0))
            
            goal_locs = np.random.choice(np.arange(len(empty_space[0])), num_agents, replace=False)
            goal_rows, goal_cols = empty_space[0][goal_locs], empty_space[1][goal_locs]
            goal_locs = np.vstack([goal_rows, goal_cols]).T
            assert(np.all(my_map[goal_rows, goal_cols] == 0))

            createScenFile(start_locs,goal_locs, map_name, f"new_map_generation/scens/{map_name}-random-{j}.scen")