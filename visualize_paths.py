import os
import matplotlib # RVMod
matplotlib.use('Agg') # RVMod
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pdb # Debugging
import argparse  # Command line arguments
import imageio.v2 as imageio  # Creating gif
import pandas as pd
from PIL import Image
import multiprocessing as mp



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

# def convertToGIF(image_folder, gif_path):
#     # Animate
#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#     images = sorted(images)

#     # Create the GIF using imageio
#     # Source: https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
#     with imageio.get_writer(gif_path, mode='I', duration=500, loop=0) as writer:
#         for image_name in images:
#             image_path = os.path.join(image_folder, image_name)
#             image = imageio.imread(image_path)
#             writer.append_data(image)
        

#     # Remove the individual image files
#     for image_name in images:
#         image_path = os.path.join(image_folder, image_name)
#         os.remove(image_path)

def create_gif(image_folder, output_path, duration=100):
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            file_path = os.path.join(image_folder, file_name)
            images.append(Image.open(file_path))

    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    else:
        print("No images found in the folder")
    
    image_names = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        os.remove(image_path)


def string_coord_to_tuple(string_coord):
    char_list = list(string_coord)
    char_list = char_list[1:]
    char_list = char_list[:-1]
    coords_str = ''.join(char_list)
    coords_str_list = coords_str.split(',')
    coords = tuple((int(coords_str_list[0]), int(coords_str_list[1])))
    return coords

def readMap(mapfile: str):
    """ Read map """
    if mapfile.startswith("../data"):
        mapfile = mapfile[3:]
    # mapfile = "data/mapf-map/random-32-32-20.map"
    # mapfile = "data/custom_tunnel2.map"
    with open(mapfile) as f:
        line = f.readline()  # type octile

        line = f.readline()  # height 32
        height = int(line.split(' ')[1])

        line = f.readline()  # width 32
        width = int(line.split(' ')[1])

        line = f.readline()  # map

        mapdata = np.array([list(line.rstrip()) for line in f])

    mapdata.reshape((width,height))
    mapdata[mapdata == '.'] = 0
    mapdata[mapdata == '@'] = 1
    mapdata[mapdata == 'T'] = 1
    mapdata = mapdata.astype(int)
    return mapdata

def animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile):
    colors = ['r', 'b', 'm', 'g']

    # Visualize
    tmpFolder = f"./{outpath}/tmpImgs_{mapname}"
    os.makedirs(tmpFolder, exist_ok=True)
    print("Animating 1 image")
    # for t in range(0, max_plan_length):
    
    last_row = id2plan[-1]
    repeated_rows = np.tile(last_row, (40,1, 1))
    id2plan = np.vstack([id2plan, repeated_rows])
    finished=False
    if np.all(id2plan[-1]==id2goal[0:agents]):
        finished=True
    for t in range(max_plan_length+40-1, -1, -1):
        plt.imshow(mapdata, cmap="Greys")
        for i in range(0, agents):
            plan = id2plan[:,i]
            if np.all(plan[t] == id2goal[i]):
                plt.scatter(plan[t][1], plan[t][0],s=3, c="grey")
            else:
                plt.scatter(plan[t][1], plan[t][0], s=3, c=colors[i%len(colors)]) # RVMod: Fixed by modding

        # if finished:
        #     plt.text(0.1, 1.15, f'success {mapname}', color='green', fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
        # else:
        #     plt.text(0.1, 1.15, f'failure {mapname}', color='red', fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
        color = 'green' if finished else 'red'
        plt.subplots_adjust(top=0.85)
        name = "{}/{:03d}.png".format(tmpFolder, t)
        plt.title(f"t = {t}", color=color)
        plt.savefig(name)
        plt.cla()
    
    succStr = "_S" if finished else "_F"
    # if outputfile is None:
    #     outputfile = "animation.gif"
    # assert(outputfile.endswith(".gif"))
    create_gif(tmpFolder, outpath+outfile+succStr+".gif")
    
def process_map(params):
    mapname, args, scen_count, shieldType = params

    # Read the map data
    mapdata = readMap(f"data_collection/data/benchmark_data/maps/{mapname}.map")
    
    # Get the directory of logs for the current map
    log_dir = f"benchmarking/{scen_count}_{shieldType}_results_full/{mapname}/paths"
    log_dir_list = os.listdir(log_dir)
    
    berlin1, berlin4, berlin16, berlin128 = False, False, False, False
    den1, den4, den16, den128 = False, False, False, False
    ware1, ware4, ware16, ware128 = False, False, False, False
    
    scen_folder = "data_collection/data/benchmark_data/scens/"
    seen = [] 
    for i, log in enumerate(log_dir_list):
        if ".npy" not in log:
            continue
        
        scen_abbr = log.split(".")[0]
        id2plan = np.load(log_dir + "/" + log)
        scen_file = scen_folder + scen_abbr + ".scen"
        start_locs, id2goal = parse_scene(scen_file)
        
        max_plan_length = id2plan.shape[0]
        print(id2plan.shape)
        agents = id2plan.shape[1]
        agent_count = log.split(".")[-2]
        success = np.all(id2plan[-1]==id2goal[0:agents])
        # key = f"{mapname}_{scen_count}_{success}_{agent_count}"
        # if key in seen:
        #     continue
        # seen.append(key)
        outpath = args.output
        outfile = f"{mapname}_s{scen_count}_a{agent_count}_{i}"
        
        if "Berlin" in mapname:
            if scen_count==1 and not berlin1 and not success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                berlin1 = True
            elif scen_count==4 and not berlin4 and success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                berlin4 = True
            elif scen_count==16 and not berlin16 and success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                berlin16 = True
            elif scen_count==128 and not berlin128 and success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                berlin128 = True
        elif "den" in mapname:
            if scen_count==1 and not den1 and not success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                den1 = True
            elif scen_count==4 and not den4 and not success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                den4 = True
            elif scen_count==16 and not den16 and success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                den16 = True
            elif scen_count==128 and not den128 and success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                den128 = True
        elif "warehouse" in mapname:
            if scen_count==1 and not ware1 and not success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                ware1 = True
            elif scen_count==4 and not ware4 and not success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                ware4 = True
            elif scen_count==16 and not ware16 and not success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                ware16 = True
            elif scen_count==128 and not ware128 and success:
                animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
                ware128 = True
        
        
        animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
        

def run_parallel_over_maps(map_list, args, scen_count, shieldType):
    # Create the list of parameters for each map
    params = [(mapname, args, scen_count, shieldType) for mapname in map_list]
    
    # Use multiprocessing Pool to parallelize processing across maps
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_map, params)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize agent paths from log file')
    # parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--output', type=str, help='Path to the output gif file', default="animations/")
    parser.add_argument('--shieldType', type=str, help='Path to the output gif file', required=True)
    parser.add_argument('--scen_count', type=int, help='Path to the output gif file', required=True)
    args = parser.parse_args()
    # map_list = ["den312d", "maze_32_32_4", "room_32_32_4", "random_32_32_10", "Berlin_1_256"]
    map_list = ["Berlin_1_256", "den312d", "warehouse_10_20_10_2_2"]
    
    run_parallel_over_maps(map_list, args, args.scen_count, args.shieldType)

if __name__ == '__main__':
    main()