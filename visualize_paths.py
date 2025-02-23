import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import argparse
import imageio.v2 as imageio
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
        for line in f:
            line = line.rstrip()
            tokens = line.split("\t")
            assert(len(tokens) == 9)
            tokens = tokens[4:]
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row, col))
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row, col))
    return np.array(start_locations, dtype=int), np.array(goal_locations, dtype=int)

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
    with open(mapfile) as f:
        line = f.readline()
        line = f.readline()
        height = int(line.split(' ')[1])
        line = f.readline()
        width = int(line.split(' ')[1])
        line = f.readline()
        mapdata = np.array([list(line.rstrip()) for line in f])

    mapdata.reshape((width, height))
    mapdata[mapdata == '.'] = 0
    mapdata[mapdata == '@'] = 1
    mapdata[mapdata == 'T'] = 1
    mapdata = mapdata.astype(int)
    return mapdata

def animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile):
    colors = ['r', 'b', 'm', 'g']
    tmpFolder = f"./{outpath}/tmpImgs_{mapname}"
    os.makedirs(tmpFolder, exist_ok=True)
    last_row = id2plan[-1]
    repeated_rows = np.tile(last_row, (40, 1, 1))
    id2plan = np.vstack([id2plan, repeated_rows])
    finished = np.all(id2plan[-1] == id2goal[0:agents])
    for t in range(max_plan_length + 40 - 1, -1, -1):
        plt.imshow(mapdata, cmap="Greys")
        for i in range(agents):
            plan = id2plan[:, i]
            if np.all(plan[t] == id2goal[i]):
                plt.scatter(plan[t][1], plan[t][0], s=3, c="grey")
            else:
                plt.scatter(plan[t][1], plan[t][0], s=3, c=colors[i % len(colors)])
        color = 'green' if finished else 'red'
        plt.subplots_adjust(top=0.85)
        name = "{}/{:03d}.png".format(tmpFolder, t)
        plt.title(f"t = {t}", color=color)
        plt.savefig(name)
        plt.cla()
    
    succStr = "_S" if finished else "_F"
    create_gif(tmpFolder, outpath + outfile + succStr + ".gif")

def process_map(params):
    mapname, args, scen_count, shieldType = params
    mapdata = readMap(f"data_collection/data/benchmark_data/maps/{mapname}.map")
    log_dir = f"benchmarking/{scen_count}_{shieldType}_results_full/{mapname}/paths"
    log_dir_list = os.listdir(log_dir)
    
    berlin1, den1, ware1 = False, False, False
    scen_folder = "data_collection/data/benchmark_data/scens/"
    for i, log in enumerate(log_dir_list):
        if ".npy" not in log:
            continue
        scen_abbr = log.split(".")[0]
        id2plan = np.load(log_dir + "/" + log)
        scen_file = scen_folder + scen_abbr + ".scen"
        start_locs, id2goal = parse_scene(scen_file)
        max_plan_length = id2plan.shape[0]
        agents = id2plan.shape[1]
        agent_count = log.split(".")[-2]
        success = np.all(id2plan[-1] == id2goal[0:agents])
        outpath = args.output
        outfile = f"{mapname}_s{scen_count}_a{agent_count}_{i}"
        
        if "Berlin" in mapname and not berlin1 and success and int(agent_count) > 800:
            animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
            berlin1 = True
        if "den" in mapname and not den1 and success and int(agent_count) > 350:
            animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
            den1 = True
        if "ware" in mapname and not ware1 and success and int(agent_count) > 900:
            animate_agents(mapname, mapdata, id2plan, id2goal, max_plan_length, agents, outpath, outfile)
            ware1 = True

def run_parallel_over_maps(map_list, args, scen_count, shieldType):
    params = [(mapname, args, scen_count, shieldType) for mapname in map_list]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_map, params)

def main():
    parser = argparse.ArgumentParser(description='Visualize agent paths from log file')
    parser.add_argument('--output', type=str, help='Path to the output gif file', default="animations/")
    parser.add_argument('--shieldType', type=str, help='Path to the output gif file', required=True)
    parser.add_argument('--scen_count', type=int, help='Path to the output gif file', required=True)
    args = parser.parse_args()
    map_list = ["Berlin_1_256", "den312d", "warehouse_10_20_10_2_2"]
    run_parallel_over_maps(map_list, args, args.scen_count, args.shieldType)

if __name__ == '__main__':
    main()
