import argparse
import numpy as np
import os
import shutil
import pdb
import time
from collections import deque
import subprocess
import csv

# generate maps with DFS backtracking
def generate_maze(height, width, corridor_size):
    """Generates a maze using DFS.

    Args:
        rows: Number of rows in the maze.
        cols: Number of columns in the maze.

    Returns:
        A list of lists representing the maze. 
        0 represents a path, and 1 represents a wall.
    """

    t0 = time.time()

    maze = np.ones((height, width))  # initialize maze with walls
    visited = np.zeros((height, width))
    moves = np.asarray([(0, 1), (1, 0), (0, -1), (-1, 0)])

    dq = deque()
    dq.append((1, 1, 1, 1)) # row, col, prev_row, prev_col
    
    while dq:
        row, col, prev_row, prev_col = dq.pop()

        if visited[row, col] == 1: continue
        visited[row, col] = 1

        maze[min(row, prev_row) : max(row + corridor_size, prev_row + corridor_size), 
             min(col, prev_col) : max(col + corridor_size, prev_col + corridor_size)] = 0

        for index in np.random.choice(range(4), size=4, replace=False):
            dr, dc = moves[index]
            new_row, new_col = row + dr * (1 + corridor_size), col + dc * (1 + corridor_size) # move two steps to leave a wall in between
            if 0 <= new_row < height and 0 <= new_col < width and not visited[new_row, new_col]:
                dq.append((new_row, new_col, row, col))

    print(f'Maze generated in {time.time() - t0:.4f}s')
    return maze


# for scen generation
# The optimal path length is assuming sqrt(2) diagonal costs.
# The optimal path length assumes agents cannot cut corners through walls
def octile_bfs(maze, start, goal):
    moves = np.asarray([(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)])
    costs = [1]*4 + [2**0.5]*4
    height, width = maze.shape

    def is_valid_move(move, new_row, new_col):
        if not (0 <= new_row < height and 0 <= new_col < width): 
            return

        corner_cut = maze[new_row - move[0], new_col] or maze[new_row, new_col - move[1]]
        return maze[new_row, new_col] == 0 and not corner_cut

    dq = deque()
    dq.append((start, 0))

    visited = np.zeros((height, width))
    visited[start[0], start[1]] = 1
    
    while dq:
        loc, cost = dq.popleft()
        
        if np.all(loc == goal):
            return cost
        
        for i in range(len(moves)):
            move, dcost = moves[i], costs[i]
            new_row, new_col = loc + move
            # check if the new position is valid and not visited
            if is_valid_move(move, new_row, new_col) and visited[new_row, new_col] == 0:
                visited[new_row, new_col] = 1
                dq.append(((new_row, new_col), cost + dcost))
    
    raise Exception("BFS should be able to find path in maze")


# permutation with elements in new positions
def derange(arr0):
    assert(len(arr0) >= 2)
    arr = np.copy(arr0)
    for a in range(1, len(arr)):
        b = np.random.choice(range(0, a))
        temp = np.copy(arr[a])
        arr[a] = arr[b]
        arr[b] = temp
    assert(all([a != b for a,b in zip(arr, arr0)]))
    return arr


def generate_scens(maze, args):
    open_locs = np.column_stack(np.where(maze == 0)) # (num_open_locs, 2)

    scen_starts = np.zeros((args.num_scens, args.num_agents, 2), dtype=int)
    scen_goals = np.zeros((args.num_scens, args.num_agents, 2), dtype=int)
    scen_costs = np.zeros((args.num_scens, args.num_agents))

    t0 = time.time()
    for scen_idx in range(args.num_scens):
        permutation = np.random.choice(range(len(open_locs)), size=len(open_locs), replace=False) # (num_open_locs,)
        start_locs = open_locs[permutation[:args.num_agents]] # (N, 2)
        goal_locs = open_locs[derange(permutation)[:args.num_agents]] # (N, 2)
        scen_starts[scen_idx, :] = start_locs
        scen_goals[scen_idx, :] = goal_locs
    print(f'Scen start goal pairs generated in {time.time() - t0:.4f}s')

    t0 = time.time()
    if args.skip_octile_bfs:
        print(f'Octile BFS skipped')
    else:
        for scen_idx in range(args.num_scens):
            for agent_idx in range(args.num_agents): 
                scen_costs[scen_idx, agent_idx] = octile_bfs(maze, scen_starts[scen_idx, agent_idx], scen_goals[scen_idx, agent_idx])
        print(f'Octile BFS completed in {time.time() - t0:.4f}s')
    
    return (scen_starts, scen_goals, scen_costs)


def save_map_file(maze, args):
    with open(f'{args.data_path}/maps/{args.maze_name}_{args.width}_{args.height}_{args.corridor_size}.map', 'w') as f:
        f.write('type octile\n')
        f.write(f'height {args.height}\n')
        f.write(f'width {args.width}\n')
        f.write('map\n')

        maze_str = '\n'.join([''.join(['@' if cell else '.' for cell in row]) for row in maze])
        f.write(maze_str)


def save_scen_files(scen_data, args):
    scen_starts, scen_goals, scen_costs = scen_data
    map_filename = f'{args.maze_name}_{args.width}_{args.height}_{args.corridor_size}'

    for scen_idx in range(args.num_scens):
        with open(f'{args.data_path}/scens/{map_filename}-random-{scen_idx+1}.scen', 'w') as f:
            f.write('version 1\n')
            # Bucket, map, map width, map height, start x-coordinate, start y-coordinate, goal x-coordinate, goal y-coordinate, optimal length
            # (0, 0) is in the upper left corner of the maps 
            for agent_idx in range(args.num_agents):
                y0, x0 = scen_starts[scen_idx, agent_idx]
                y1, x1 = scen_goals[scen_idx, agent_idx]
                cost = scen_costs[scen_idx, agent_idx]
                
                scen_row_str = '\t'.join(map(str, [0, map_filename+'.map', args.width, args.height,
                                          x0, y0, x1, y1, f'{cost:.8f}']))
                f.write(scen_row_str + '\n')


def generate_map_scens(args):
    # generate map
    maze = generate_maze(args.height, args.width, args.corridor_size)
    num_open_locs = int(args.width*args.height-np.sum(maze))
    assert(num_open_locs > 1)
    args.num_agents = min(args.num_agents, num_open_locs)

    # generate scens
    scen_data = generate_scens(maze, args)
    
    # output files
    save_map_file(maze, args)
    save_scen_files(scen_data, args)


def generate_constants(args):
    constants_generator_cmd = f'''python -m data_collection.constants_generator --mapFolder={args.data_path}/maps \
        --scenFolder={args.data_path}/scens \
        --constantMapAndBDFolder={args.data_path}/constant_npzs \
        --outputFolder={args.temp_bd_path}/ \
        --num_parallel_runs=1 \
        "eecbs" \
        --eecbsPath={args.eecbs_path} \
        --firstIter=true --cutoffTime=1'''# \
        # --deleteTextFiles=true '''s
    subprocess.run(constants_generator_cmd, shell=True, check=True)
    

'''
maze_config_csv: 
maze_name,height,width,corridor_size,num_agents,num_scens
maze1,16,16,1,1000,25

Example runs:
python -m data_collection.maze_generator --data_path=$PROJECT/data/maze_benchmark_data/ \
        --temp_bd_path=$PROJECT/data/logs/EXP_Generate_mazes/ \
        --maze_config_csv=$PROJECT/data/mazes_test.csv \
        --eecbs_path=./data_collection/eecbs/build_release5/eecbs --skip_octile_bfs

python -m data_collection.maze_generator --data_path=data_collection/data/maze_benchmark_data/ \
        --temp_bd_path=data_collection/data/logs/EXP_Generate_mazes/ \
        --maze_config_csv=data_collection/data/mazes_test.csv \
        --eecbs_path=./data_collection/eecbs/build_release5/eecbs --skip_octile_bfs
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--maze_config_csv', type=str, help='mazes to generate', required=True)
    parser.add_argument('--data_path', type=str, help='name of folder with data', required=True)
    parser.add_argument('--temp_bd_path', type=str, help='temp paths/ and csvs/ path for constants_generator.py if generating constants', default='')
    parser.add_argument('--eecbs_path', type=str, help='eecbs path for constants_generator.py if generating constants', default='')
    parser.add_argument('--skip_octile_bfs', action='store_true')

    args = args = parser.parse_args()

    np.random.seed(0)
    
    # make data directories
    try:
        shutil.rmtree(args.data_path)
    except: pass
    
    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.data_path+'/maps', exist_ok=True)
    os.makedirs(args.data_path+'/scens', exist_ok=True)

    with open(args.maze_config_csv, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            values = [row[0]] + [*map(int, row[1:])]
            map_args = argparse.Namespace(**dict(zip(header, values)))
            map_args.data_path = args.data_path
            map_args.skip_octile_bfs = args.skip_octile_bfs
            generate_map_scens(map_args)
    print(f'Maps and scens in {args.data_path}')

    if args.eecbs_path and args.temp_bd_path:
        print('Running constants_generator.py...')
        generate_constants(args)
    
    print(f'Done.\n')


