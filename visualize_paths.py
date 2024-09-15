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

def create_gif(image_folder, output_path, duration=60):
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

def animate_agents(mapdata, id2plan, id2goal, max_plan_length, agents, outputfile):
    colors = ['r', 'b', 'm', 'g']

    # Visualize
    tmpFolder = "./animations/tmpImgs"
    print("Animating 1 image")
    # for t in range(0, max_plan_length):
    
    last_row = id2plan[-1]
    repeated_rows = np.tile(last_row, (40,1, 1))
    id2plan = np.vstack([id2plan, repeated_rows])
    finished=False
    if np.all(id2plan[-1]==id2goal[0:agents]):
        finished=True
        print("successful run skipping")
        return
    for t in range(max_plan_length+40-1, -1, -1):
        plt.imshow(mapdata, cmap="Greys")
        for i in range(0, agents):
            plan = id2plan[:,i]
            if np.all(plan[t] == id2goal[i]):
                plt.scatter(plan[t][1], plan[t][0],s=1, c="grey")
            else:
                plt.scatter(plan[t][1], plan[t][0], s=1, c=colors[i%len(colors)]) # RVMod: Fixed by modding

        if finished:
            plt.text(0.2, 1.05, 'success', color='green', fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.2, 1.05, 'failure', color='red', fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
        plt.subplots_adjust(top=0.85)
        name = "{}/{:03d}.png".format(tmpFolder, t)
        plt.title(f"t = {t}")
        plt.savefig(name)
        plt.cla()
    
    succStr = "_S" if finished else "_F"
    # if outputfile is None:
    #     outputfile = "animation.gif"
    # assert(outputfile.endswith(".gif"))
    create_gif(tmpFolder, outputfile+succStr+".gif")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize agent paths from log file')
    # parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--output', type=str, help='Path to the output gif file', default="animations/den312d")
    args = parser.parse_args()

    # Call the animate_agents function with the log file path
    mapdata = readMap("data_collection/data/benchmark_data/maps/den312d.map")
    # log_file = "data_collection/data/logs/EXP_Medium_4/iter4/pymodel_outputs/random_32_32_10/paths/random_32_32_10-random-1.random_32_32_10-random-1.npy"
    # scen_file = "data_collection/data/logs/EXP_Medium_4/iter4/pymodel_outputs/random_32_32_10/paths/random_32_32_10-random-1.random_32_32_10-random-1_t13.100.scen"
    log_dir = "benchmarking/16_CSFreeze_results_full/den312d/paths"
    log_dir_list = os.listdir(log_dir)
    def process_log(params):
        i, log, log_dir, scen_folder, args = params
        if ".npy" not in log:
            return
        
        scen_abbr = log.split(".")[0]
        id2plan = np.load(log_dir + log)
        scen_file = scen_folder + scen_abbr + ".scen"
        start_locs, id2goal = parse_scene(scen_file)
        
        max_plan_length = id2plan.shape[0]
        print(id2plan.shape)
        agents = id2plan.shape[1]
        
        output = f"{args.output}_{i}"
        animate_agents(mapdata, id2plan, id2goal, max_plan_length, agents, output)

    def run_in_parallel(log_dir_list, log_dir, args):
        scen_folder = "data_collection/data/benchmark_data/scens/"
        
        # Create a list of parameters for each task
        params = [(i, log, log_dir, scen_folder, args) for i, log in enumerate(log_dir_list)]

        # Use multiprocessing Pool to parallelize the process_log function
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(process_log, params)
        
    run_in_parallel(log_dir_list, log_dir, args)
    

    
    # for i, log in enumerate(log_dir_list):
    #     if ".npy" not in log:
    #         continue
    #     scen_abbr = log.split(".")[0]
    #     # index = [idx for idx, s in enumerate(log_dir_list) if scen_abbr in s and ".npy" not in s and "400" in s][0]
    #     scen_folder = "data_collection/data/benchmark_data/scens/"
    #     id2plan = np.load(log_dir + log)
    #     scen_file = scen_folder+scen_abbr+".scen"
    #     start_locs, id2goal  = parse_scene(scen_file)
        
    #     max_plan_length = id2plan.shape[0]
    #     print(id2plan.shape)
    #     agents = id2plan.shape[1]
    #     # mapdata, id2plan, id2goal, max_plan_length, agents = readJSSSolutionPaths(args.log_file)
    #     output = f"{args.output}_{i}"
    #     animate_agents(mapdata, id2plan, id2goal, max_plan_length, agents, output)



def plotAgentsOnMap(ax, id2plan, max_plan_length, mapdata, t, getColor):
    """ Plot agents on the map    
    """
    # Visualize
    ax.imshow(mapdata, cmap="Greys")
    for i in id2plan.keys():
        plan = id2plan[i]
        if t > len(plan)-1:
            ax.scatter(plan[-1][1], plan[-1][0], c=getColor(i)) # RVMod: Fixed by modding
        else:
            ax.scatter(plan[t][1], plan[t][0], c=getColor(i)) # RVMod: Fixed by modding
    # name = "{}/{:03d}.png".format(tmpFolder, t)
    # ax.set_title(f"t = {t}")


def createSingleFrame(id2plan, max_plan_length, mapdata, t, df):
    def parseSubgroup(all_subgroups_str):
        if (str(all_subgroups_str) == "nan"):
            return []
        per_subgroup_string = all_subgroups_str.split(";")
        subgroups_parsed = []
        for subgroup in per_subgroup_string:
            agents = subgroup[1:-2].split("-")
            agents = set([int(agent) for agent in agents])
            subgroups_parsed.append(agents)
        return subgroups_parsed
    
    subgroups = parseSubgroup(df["agent_subgroups"][t])
    if str(df["hp_penalty_vals"][t]) == "nan":
        hp_penalties = np.array([])
    else:
        hp_penalties = np.array(df["hp_penalty_vals"][t].split(";"), dtype=float).astype(int)

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6+4,nrows*6))
    flat_axes = axes.flatten()

    ## Plot current position of agents
    curAx = flat_axes[0]
    def getAgentColor(i):
        colors = ['r', 'b', 'm', 'g']
        return colors[i%len(colors)]
    plotAgentsOnMap(curAx, id2plan, max_plan_length, mapdata, t, getAgentColor)

    ## Plot next position of agents with subgroup coloring
    curAx = flat_axes[1]
    allAgentsInSubgroups = []
    for subgroup in subgroups:
        allAgentsInSubgroups.extend(subgroup)
    allAgentsInSubgroups = set(allAgentsInSubgroups)
    def getSubgroupColor(i):
        if i in allAgentsInSubgroups:
            for j, subgroup in enumerate(subgroups):
                if i in subgroup:
                    return getAgentColor(min(subgroup))
        else:
            return 'grey'
    plotAgentsOnMap(curAx, id2plan, max_plan_length, mapdata, t+1, getSubgroupColor)

    # ## Showing the sum of g_val, h_val, and hp_val
    g_val = df["g_val"][:t+1].sum()
    h_val = df["h_val"][t]
    hp_val = df["hp_val"][t]
    curAx.set_title("{:d} = {:d} + {:d} + {:d}".format(g_val+h_val+hp_val, g_val, h_val, hp_val))

    ## Showing the sum of g_val, h_val, and hp_val
    # pdb.set_trace()
    # g_val = df["g_val"][t]
    # h_val = df["h_val"][t]
    # hp_val = df["hp_val"][t]
    # total_h_val = h_val + hp_val
    # if t == max_plan_length - 2: # Note 2 and not 1 as if we only had 2 frames, we could only plot t=0
    #     next_h_val = 0
    #     next_hp_val = 0
    # else:
    #     next_h_val = df["h_val"][t+1]
    #     next_hp_val = df["hp_val"][t+1]
    # next_total_h_val = next_h_val + next_hp_val
    # curAx.set_title("{:d} = {:d} + {:d} - {:d}".format(g_val + next_total_h_val - total_h_val, g_val, next_total_h_val, total_h_val))

    ## Plot histogram of g_vals and h_vals
    curAx = flat_axes[2]
    xAxis = list(range(0, t+1))
    curYVals = 0
    colors = ['r', 'b', 'm', 'g']
    for i, key in enumerate(["hp_val", "h_val", "g_val"]):
        color = colors[i]
        vals = df[key][:t+1]
        if key == "g_val":
            # vals = vals.cumsum()
            pass
        curAx.plot(curYVals+vals, label=key, color=color, alpha=0.5, marker='o')
        curAx.fill_between(xAxis, curYVals, curYVals+vals, color=color, alpha=0.5)
        curYVals += vals
    curAx.set_yscale('symlog')
    curAx.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

    # nextVal = df["g_val"][t+1]
    # curAx.plot(nextVal, color='blue', label="g_val")
    # curAx.fill_between(xAxis, 0, nextVal, color='blue', alpha=0.5)
    # nextVal += df["h_val"][t+1]
    # curAx.plot(df["g_val"][:t+1]+df["h_val"][:t+1], label="h_val")
    # curAx.plot(df["g_val"][:t+1]+df["h_val"][:t+1]+df["hp_val"][:t+1], label="hp_val")


    ## Plot histogram of hp_penalties
    curAx = flat_axes[3]
    curAx.set_title("Histogram of HP penalties")
    curAx.set_xlabel("HP penalties")
    curAx.set_ylabel("Frequency")
    if len(hp_penalties) != 0:
        bins = np.arange(hp_penalties.min(), hp_penalties.max()+2, 1) - 0.5
        curAx.hist(hp_penalties, bins=bins, color='blue', edgecolor='black', alpha=0.8)
        curAx.set_xticks(np.arange(hp_penalties.min(), hp_penalties.max() + 1, 1))

# def createFancyAnimation():
#     names = ["den520d_300", "random_80"]
#     curName = names[0]
#     # paths_path = "/home/rishi/research/eecbs-private/build_debug/jss_solution_paths.txt"
#     # animation_stats_path = "/home/rishi/research/eecbs-private/build_debug/animation_stats.csv"
#     paths_path = f"/home/rishi/research/eecbs-private/build_release/{curName}_animation.txt"
#     animation_stats_path = f"/home/rishi/research/eecbs-private/build_release/{curName}_animation.csv"
    
#     # max_plan_length = max(id2length.values())
#     mapdata, id2plan, id2goal, max_plan_length, agents = readJSSSolutionPaths(paths_path)

#     # Read animation stats
#     df = pd.read_csv(animation_stats_path)
#     # pdb.set_trace()
#     df["hp_penalty_vals"] = df["hp_penalty_vals"].astype(str)

#     # Visualize
#     tmpFolder = "data/logs/tmpImages"
#     print('Total frames: ', max_plan_length-1)
#     # max_plan_length = 10
#     for t in range(0, max_plan_length-1):
#         createSingleFrame(id2plan, max_plan_length, mapdata, t, df)
#         name = "{}/{:03d}.png".format(tmpFolder, t)
#         plt.savefig(name, bbox_inches='tight')
#         # plt.cla()
#         plt.close()
#         print(t)

#     convertToGIF(tmpFolder, f"{curName}_fancy.gif")

if __name__ == '__main__':
    # main()
    main()