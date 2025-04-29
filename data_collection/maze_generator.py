import argparse
import numpy as np
import os
import shutil
import pdb
import time
from collections import deque
import subprocess
import csv
import random
from tqdm import tqdm

# generate maze map with DFS backtracking
def generate_maze(height, width, corridor_size):
    """Generates a maze map using DFS.
        height: Number of rows in the maze
        width: Number of columns in the maze
        corridor_size: width of space between walls
    """

    maze = np.ones((height, width))  # initialize map with walls
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

    return maze


# generate room maze map with DFS backtracking
def generate_room_maze(height, width, corridor_size, room_size):
    """Generates a maze map using DFS.
        height: Number of rows in the maze
        width: Number of columns in the maze
        corridor_size: width of corridor between rooms
        room_size: width of room (square)
    """
    
    maze = np.ones((height, width))  # initialize map with walls
    visited = np.zeros((height, width))
    moves = np.asarray([(0, 1), (1, 0), (0, -1), (-1, 0)])

    dq = deque()
    dq.append((1, 1, 1, 1)) # row, col, prev_row, prev_col
    
    while dq:
        row, col, prev_row, prev_col = dq.pop()

        if visited[row, col] == 1: continue
        visited[row, col] = 1

        maze[row : row + room_size, col : col + room_size] = 0
        if prev_row != row:
            random_col = random.randint(prev_col, prev_col + room_size - corridor_size)
            maze[min(prev_row, row) : max(prev_row + room_size, row + room_size), 
                 random_col : random_col + corridor_size] = 0
        else:
            random_row = random.randint(prev_row, prev_row + room_size - corridor_size)
            maze[random_row : random_row + corridor_size, 
                 min(prev_col, col) : max(prev_col + room_size, col + room_size)] = 0

        for index in np.random.choice(range(4), size=4, replace=False):
            dr, dc = moves[index]
            new_row, new_col = row + dr * (1 + room_size), col + dc * (1 + room_size) # move two steps to leave a wall in between
            if 0 <= new_row < height and 0 <= new_col < width and not visited[new_row, new_col]:
                dq.append((new_row, new_col, row, col))

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
    
    return -1
    # raise Exception("BFS should be able to find path in maze")


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


def generate_scens(maze, maze_filename, args):
    open_locs = np.column_stack(np.where(maze == 0)) # (num_open_locs, 2)

    if args.num_agents == -1:
        args.num_agents = len(open_locs)-1
    else:
        args.num_agents = min(args.num_agents, len(open_locs)-1)
        
    scen_starts = np.zeros((args.num_scens, args.num_agents, 2), dtype=int)
    scen_goals = np.zeros((args.num_scens, args.num_agents, 2), dtype=int)
    scen_costs = np.zeros((args.num_scens, args.num_agents))

    regions = find_connected_regions(maze)
    print('Regions:', [len(s) for s in regions])
    for scen_idx in range(args.num_scens):
        start_locs, goal_locs = get_start_goal_locs(args.scen_type, open_locs, args.num_agents, args.width, args.height, args.use_existing_maps, regions)
        scen_starts[scen_idx, :] = start_locs
        scen_goals[scen_idx, :] = goal_locs

    if not args.skip_octile_bfs:
        for scen_idx in range(args.num_scens):
            for agent_idx in range(args.num_agents): 
                scen_costs[scen_idx, agent_idx] = octile_bfs(maze, scen_starts[scen_idx, agent_idx], scen_goals[scen_idx, agent_idx])

    save_scen_files((scen_starts, scen_goals, scen_costs), maze_filename, args)


def find_connected_regions(maze):
    rows, cols = len(maze), len(maze[0])
    visited = [[False] * cols for _ in range(rows)]
    regions = []

    def dfs_iterative(start_r, start_c):
        region = set()
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if (r < 0 or c < 0 or r >= rows or c >= cols or 
                maze[r][c] == 1 or visited[r][c]):
                continue
            visited[r][c] = True
            region.add((r, c))
            # Add 4-directional neighbors
            stack.append((r + 1, c))
            stack.append((r - 1, c))
            stack.append((r, c + 1))
            stack.append((r, c - 1))
        return region

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0 and not visited[r][c]:
                region = dfs_iterative(r, c)
                regions.append(region)

    return regions


def get_start_goal_locs(scen_type, open_locs, num_agents, width, height, use_existing_maps, regions):
    if scen_type == 'random' and use_existing_maps:
        start_locs = []
        goal_locs = []
        start_locs_choices = list(map(tuple, open_locs))
        goal_locs_choices = list(map(tuple, open_locs))
        for _ in tqdm(range(num_agents)):
            start_idx = np.random.choice(range(len(start_locs_choices)), 1)[0]
            start_loc = start_locs_choices[start_idx]
            assert(start_loc not in start_locs)
            for region in regions:
                if start_loc in region:
                    goal_loc_region_choices = set.copy(region)
                    goal_loc_region_choices.remove(start_loc)
                    goal_loc_region_choices &= set(goal_locs_choices)
                    goal_loc_region_choices = list(goal_loc_region_choices)
                    break
            goal_idx = np.random.choice(range(len(goal_loc_region_choices)), size=1)[0]
            goal_loc = goal_loc_region_choices[goal_idx]
            assert(goal_loc not in goal_locs)
            assert(goal_loc != start_loc)
            assert(start_loc in region and goal_loc in region)
            
            start_locs.append(start_loc)
            goal_locs.append(goal_loc)

            start_locs_choices.remove(start_loc)
            goal_locs_choices.remove(goal_loc)
        start_locs = np.asarray(start_locs)   
        goal_locs = np.asarray(goal_locs)       

    elif scen_type == 'random':
        permutation = np.random.choice(range(len(open_locs)), size=len(open_locs), replace=False) # (num_open_locs,)
        start_locs = open_locs[permutation[:num_agents]] # (N, 2)
        goal_locs = open_locs[derange(permutation)[:num_agents]] # (N, 2)

    elif scen_type == 'cluster':
        # pick 2 random points on opposite sides of the map
        if random.randint(0, 1):
            rand_h = random.randint(0, height-1)
            p1 = (0, rand_h) # left
            p2 = (width-1, height-1-rand_h) # right
        else:
            rand_w = random.randint(0, width-1)
            p1 = (rand_w, 0) # top
            p2 = (width-1-rand_w, height-1) # bottom
        if random.randint(0, 1):
            start_center = p1
            goal_center = p2
        else:
            start_center = p2
            goal_center = p1

        start_locs = open_locs[np.argsort(np.sum((open_locs - start_center) ** 2, axis=1))][:num_agents]
        goal_locs = open_locs[np.argsort(np.sum((open_locs - goal_center) ** 2, axis=1))][:num_agents]

    return start_locs, goal_locs


def save_map_file(maze, args):
    if args.room_size == -1:
        room_str = ''
    else:
        room_str = f'_{args.room_size}'
    maze_name = f'{args.name}_{args.width}_{args.height}_{args.corridor_size}{room_str}'
    with open(f'{args.data_path}/maps/{maze_name}.map', 'w') as f:
        f.write('type octile\n')
        f.write(f'height {args.height}\n')
        f.write(f'width {args.width}\n')
        f.write('map\n')

        map_str = '\n'.join([''.join(['@' if cell else '.' for cell in row]) for row in maze])
        f.write(map_str)
    return maze_name


def save_scen_files(scen_data, maze_name, args):
    scen_starts, scen_goals, scen_costs = scen_data

    for scen_idx in range(args.num_scens):
        with open(f'{args.data_path}/scens/{maze_name}-{args.scen_type}-{scen_idx+1}.scen', 'w') as f:
            f.write('version 1\n')
            # Bucket, map, map width, map height, start x-coordinate, start y-coordinate, goal x-coordinate, goal y-coordinate, optimal length
            # (0, 0) is in the upper left corner of the maps 
            for agent_idx in range(args.num_agents):
                y0, x0 = scen_starts[scen_idx, agent_idx]
                y1, x1 = scen_goals[scen_idx, agent_idx]
                cost = scen_costs[scen_idx, agent_idx]
                
                scen_row_str = '\t'.join(map(str, [0, maze_name+'.map', args.width, args.height,
                                          x0, y0, x1, y1, f'{cost:.8f}']))
                f.write(scen_row_str + '\n')


def generate_map(args):
    # generate map
    if args.map_type == 'maze':
        maze = generate_maze(args.height, args.width, args.corridor_size)
    elif args.map_type == 'room':
        maze = generate_room_maze(args.height, args.width, args.corridor_size, args.room_size)

    num_open_locs = int(args.width*args.height-np.sum(maze))
    assert(num_open_locs > 1)
    args.num_agents = min(args.num_agents, num_open_locs)
    
    # output files
    maze_name = save_map_file(maze, args)
    return maze, maze_name


def parse_octile_map(map_str):
    lines = map_str.strip().split('\n')
    start_index = lines.index('map') + 1
    map_lines = lines[start_index:]

    height = len(map_lines)
    width = len(map_lines[0])
    grid = np.zeros((height, width), dtype=int)

    for i, line in enumerate(map_lines):
        for j, char in enumerate(line):
            if char != '.':
                grid[i, j] = 1

    return grid


'''
maze_config_csv: 
name,map_type,scen_type,height,width,corridor_size,room_size,num_agents,num_scens

New scens for old maps:
python -m data_collection.maze_generator --data_path=data_collection/data/benchmark_data_eval/ --use_existing_maps --skip_octile_bfs --skip_derange

Small run:
python -m data_collection.maze_generator --data_path=data_collection/data/mini_maze_benchmark_data/ --skip_octile_bfs
python -m data_collection.maze_generator --data_path=$PROJECT/data/mini_maze_benchmark_data/ --skip_octile_bfs

Full run:
python -m data_collection.maze_generator --data_path=$PROJECT/data/maze_benchmark_data/
        --num_maps_per_type=2 \
        --min_size=16 --max_size=64 \
        --min_corridor_size=1 --max_corridor_size=3 \
        --min_room_size=3 --max_room_size=5 \
        --num_scens_per_map=25 --skip_octile_bfs

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='name of folder with maps and scens', required=True)
    
    parser.add_argument('--skip_octile_bfs', action='store_true')

    # map config
    parser.add_argument('--num_maps_per_type', type=int, help='num maps of each type to generate', default=1)
    parser.add_argument('--min_size', type=int, default=16)
    parser.add_argument('--max_size', type=int, default=32)
    parser.add_argument('--min_corridor_size', type=int, default=1)
    parser.add_argument('--max_corridor_size', type=int, default=1)
    parser.add_argument('--min_room_size', type=int, default=3)
    parser.add_argument('--max_room_size', type=int, default=3)
    parser.add_argument('--num_scens_per_map', type=int, default=2)

    parser.add_argument('--use_existing_maps', action='store_true')
    parser.add_argument('--skip_derange', action='store_true')

    args = args = parser.parse_args()

    np.random.seed(0)

    if args.use_existing_maps:
        maps = os.listdir(args.data_path+'/maps')
        try:
            shutil.rmtree(args.data_path+'/scens')
        except: pass
        os.makedirs(args.data_path+'/scens', exist_ok=True)

        for map_name in maps:
            maze_name = os.path.basename(map_name)[:-4]
            with open(f'{args.data_path}/maps/{map_name}', 'r') as f:
                maze = parse_octile_map(f.read())
            map_args = {'name': maze_name,
                        'scen_type': random,
                        'height': maze.shape[0],
                        'width': maze.shape[1],
                        'num_agents': 1000,
                        'num_scens': args.num_scens_per_map,
            }
            map_args = argparse.Namespace(**dict(zip(map_args.keys(), map_args.values())))
            map_args.scen_type = 'random'
            map_args.data_path = args.data_path
            map_args.skip_octile_bfs = args.skip_octile_bfs
            map_args.use_existing_maps = args.use_existing_maps
            generate_scens(maze, maze_name, map_args)
            print(f' {args.num_scens_per_map} random scens')
        exit(0)
    
    # make data directories
    try:
        shutil.rmtree(args.data_path)
    except: pass

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.data_path+'/maps', exist_ok=True)
    os.makedirs(args.data_path+'/scens', exist_ok=True)

    map_sizes = []
    s = 1
    while s <= args.max_size:
        if s >= args.min_size:
            map_sizes.append(s)
        s *= 2

    for map_type in ['maze', 'room']:
        for size in map_sizes:
            for corridor_size in range(args.min_corridor_size, args.max_corridor_size+1):
                if map_type == 'room':
                    room_sizes = range(args.min_room_size, args.max_room_size+1)
                else:
                    room_sizes = [-1]
                for room_size in room_sizes:
                    if map_type == 'room':
                        adjusted_size = ((size-1)//(room_size+1)+1)*(room_size+1)+1
                        assert(corridor_size <= room_size)
                    else:
                        adjusted_size = ((size-1)//(corridor_size+1)+1)*(corridor_size+1)+1
                    for n in range(args.num_maps_per_type):
                        map_args = {'name': f'new_{map_type}_{n}',
                                    'map_type': map_type,
                                    'scen_type': None,
                                    'height': adjusted_size,
                                    'width': adjusted_size,
                                    'corridor_size': corridor_size,
                                    'room_size': room_size,
                                    'num_agents': 1000,
                                    'num_scens': args.num_scens_per_map,
                        }
                        print(f'Generatiing {adjusted_size}x{adjusted_size} {map_type} map with corridor size {corridor_size} and room size {room_size}')
                        map_args = argparse.Namespace(**dict(zip(map_args.keys(), map_args.values())))
                        map_args.data_path = args.data_path
                        map_args.skip_octile_bfs = args.skip_octile_bfs
                        maze, maze_name = generate_map(map_args)
                        for scen_type in ['random', 'cluster']:
                            map_args.scen_type = scen_type
                            generate_scens(maze, maze_name, map_args, skip_derange=args.skip_derange)
                            print(f' {args.num_scens_per_map} {scen_type} scens')

    print(f'Maps and scens in {args.data_path}')
    print(f'Done.\n')


