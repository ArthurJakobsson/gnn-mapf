import os
import argparse
import pdb
import numpy as np
import torch # For the model
import pandas as pd # For saving the results
import csv # For saving the results

from gnn.dataloader import create_data_object, normalize_graph_data
from gnn.trainer import GNNStack, CustomConv # Required for using the model even if not explictly called

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
        sep = "\t"
        for line in f:
            line = line.rstrip() #remove the new line
            tokens = line.split(sep)
            # num_of_cols = int(tokens[2])
            # num_of_rows = int(tokens[3])
            tokens = tokens[4:]  # Skip the first four elements
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row,col))
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row,col))
    return np.array(start_locations, dtype=int), np.array(goal_locations, dtype=int)

def convertProbsToPreferences(probs, conversion_type):
    """Converts probabilities to preferences
    Inputs:
        probs: (N,5) probabilities
        conversion_type: sorted or sampled
    Outputs:
        preferences: (N,5) preferences with each row containing 0,1,2,3,4
    """
    if conversion_type == "sorted":
        preferences = np.argsort(-probs, axis=1)
    elif conversion_type == "sampled":
        preferences = np.zeros_like(probs, dtype=int)
        for i in range(probs.shape[0]):
            preferences[i] = np.random.choice(5, size=5, replace=False, p=probs[i])
    else:
        raise ValueError('Invalid conversion type: {}'.format(conversion_type))
    return preferences

def getCosts(solution_path, goal_locs):
    """
    Inputs:
        solution_path: (T,N,2)
        goal_locs: (N,2)
    Outputs:
        total_cost_true: sum of true costs (intermediate waiting at goal incurs costs)
        total_cost_not_resting_at_goal: sum of costs if not resting at goal
    """
    print(solution_path.shape)
    at_goal = np.all(np.equal(solution_path, np.expand_dims(goal_locs, 0)), axis=2) # (T,N,2), (1,N,2) -> (T,N)

    # Find the last timestep each agent is not at the goal
    not_at_goal = 1 - at_goal # (T,N)
    last_timestep_at_goal = len(at_goal) - np.argmax(not_at_goal[::-1], axis=0) # (N), argmax returns first occurence, so reverse
    last_timestep_at_goal = np.minimum(last_timestep_at_goal, len(at_goal)-1) # Agents that never reach goal will be fixed here
    total_cost_true = last_timestep_at_goal.sum()

    resting_at_goal = np.logical_and(at_goal[:-1], at_goal[1:]) # (T-1,N)
    total_cost_not_resting_at_goal = (1-resting_at_goal).sum()

    num_agents_at_goal = np.sum(at_goal[-1])
    assert(total_cost_true >= total_cost_not_resting_at_goal)
    assert(total_cost_true <= (solution_path.shape[0]-1)*solution_path.shape[1])
    return total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal

def testGetCosts():
    goal_locs = np.array([[1,10], [2,20], [3,30]])
    solution_path = np.array([
        [[1,9],  [1,10], [1,10], [1,9],  [1,10]], # TC=4, TNRAG=3
        [[0,20], [1,20], [2,20], [2,20], [2,20]], # TC=2, TNRAG=2
        [[5,4],  [5,4],  [5,4],  [5,4],  [5,4]]   # TC=4, TNRAG=4
    ]) # (N=3,T=5,2)
    solution_path = np.swapaxes(solution_path, 0, 1) # (T,N,2)
    # pdb.set_trace()

    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal = getCosts(solution_path, goal_locs)
    assert(total_cost_true == 4+2+4)
    assert(total_cost_not_resting_at_goal == 3+2+4)
    assert(num_agents_at_goal == 2)
    print("getCosts test passed!")


def simulate(model, k, m, grid_map, bd, start_locations, goal_locations, 
             max_steps, shield_type, seed):
    cur_locs = start_locations # (N,2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent_priorities = np.random.permutation(len(cur_locs)) # (N)
    solution_path = [cur_locs.copy()]
    success = False
    for step in range(max_steps):
        # Create the data object
        data = create_data_object(cur_locs, bd, grid_map, k, m)
        data = normalize_graph_data(data, k)
        data = data.to(device)

        # Forward pass
        _, predictions = model(data)
        probabilities = torch.softmax(predictions, dim=1) 

        # Get the action preferences
        probs = probabilities.cpu().detach().numpy() # (N,5)
        action_preferences = convertProbsToPreferences(probs, "sampled") # (N,5)
        # pdb.set_trace()

        # Run the shield
        new_move, cspibt_worked = lacam(shield_type, grid_map, action_preferences, cur_locs, 
                                        agent_priorities, [])
        if not cspibt_worked:
            raise RuntimeError('CS-PIBT failed; should never fail when no using LaCAM constraints!')
        # cur_locs = cur_locs + new_move # (N,2)
        cur_locs += new_move # (N,2)
        solution_path.append(cur_locs.copy())

        # Check if all agents have reached their goals
        if np.all(np.equal(cur_locs, goal_locations)):
            success = True
            break
    
    # pdb.set_trace()
    solution_path = np.array(solution_path) # (T<=max_steps+1,N,2)
    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal = getCosts(solution_path, goal_locations)

    return solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success


def pibt(grid_map, agent_id, action_preferences, planned_agents, move_matrix, 
         occupied_nodes, occupied_edges, current_locs, current_locs_to_agent,
         constrained_agents):
    """Inputs:
        grid_map: (H,W)
        agent_id: int
        action_preferences: (N,5)
        planned_agents: list of agent_ids
        move_matrix: (N,2)
        occupied_nodes: set (row, col)
        occupied_edges: set (row_from, col_from, row_to, col_to)
        current_locs: (N,2)
        current_locs_to_agent: dict: (row, col) -> agent_id
        constrained_agents: [(agent_id, action index), ...], empty list if no constraints
    """
    def findAgentAtLocation(aLoc):
        if tuple(aLoc) in current_locs_to_agent.keys():
            return current_locs_to_agent[tuple(aLoc)]
        else:
            return -1

    up = [-1, 0]
    down = [1, 0]
    left = [0, -1]
    right = [0, 1]
    stop = [0, 0]    
    moves_ordered = np.array([up, left, down, right, stop])

    moves_ordered = moves_ordered[action_preferences[agent_id]]
    if agent_id in constrained_agents.keys(): # Force agent to only pick that action if constrained
        action_index = constrained_agents[agent_id]
        moves_ordered = moves_ordered[action_index:action_index+1]

    current_pos = current_locs[agent_id] # (2)
    for aMove in moves_ordered:
        next_loc = current_pos + aMove # (2)
        
        # Skip if would leave map bounds
        if next_loc[0] < 0 or next_loc[0] >= grid_map.shape[0] or next_loc[1] < 0 or next_loc[1] >= grid_map.shape[1]:
            continue
        # Skip if obstacle
        if grid_map[next_loc[0], next_loc[1]] == 1:
            continue
        # Skip if vertex occupied by higher agent
        if tuple(next_loc) in occupied_nodes:
            continue
        # Skip if reverse edge occupied by higher agent
        if tuple([*next_loc, *current_pos]) in occupied_edges:
            continue
        
        ### Pretend we move there
        move_matrix[agent_id] = aMove
        planned_agents.append(agent_id)
        occupied_nodes.append(tuple(next_loc))
        occupied_edges.append(tuple([*current_pos, *next_loc]))
        conflicting_agent = findAgentAtLocation(next_loc)
        if conflicting_agent != -1 and conflicting_agent != agent_id and conflicting_agent not in planned_agents:
            # Recurse
            isvalid = pibt(grid_map, conflicting_agent, action_preferences, planned_agents,
                                move_matrix, occupied_nodes, occupied_edges, current_locs,
                                current_locs_to_agent, constrained_agents)
            if isvalid:
                return True
            else:
                del planned_agents[-1]
                del occupied_nodes[-1]
                del occupied_edges[-1]
                continue
        else:
            # No conflict
            return True
        
    # No valid move found
    return False


def lacam(lacam_or_cspibt, grid_map, action_preferences, current_locs, 
          agent_priorities, agent_constraints):
    '''
    Runs LaCAM or PIBT
    Args:
        lacam_or_pibt: "CS-PIBT" or "LaCAM"
        actionPreds: (N,5) action preds
        current_locs: (N,2) current positions
        agent_priorities: (N) agent priorities
        agent_constraints: [(agent_id, action index), ...], empty list if no constraints
    Returns:
        new_move: valid move without collision (N,2)
    '''
    assert(lacam_or_cspibt in ["CS-PIBT", "LaCAM"])
    agent_order = np.argsort(-agent_priorities) # Sort by priority, highest first
    move_matrix = np.zeros((len(agent_priorities), 2), dtype=int) # (N,2)
    occupied_nodes = []
    occupied_edges = []
    planned_agents = []

    current_locs_to_agent = dict()
    for agent_id in agent_order:
        if tuple(current_locs[agent_id]) not in current_locs_to_agent.keys():
            current_locs_to_agent[tuple(current_locs[agent_id])] = agent_id
        else:
            print("UH OH, MULTIPLE AGENTS AT SAME LOCATION!")
            pdb.set_trace()

    ### Plan constrained agents first
    ## Convert agent_id constraints to actual agent indices based on agent_order
    constrained_agents = dict()
    for agent_id, action_index in agent_constraints:
        which_agent = agent_order[agent_id]
        constrained_agents[which_agent] = action_index

    cspibt_worked = True
    for agent_id in agent_order:
        if agent_id in planned_agents:
            continue
        cspibt_worked = pibt(grid_map, agent_id, action_preferences, planned_agents, 
                            move_matrix, occupied_nodes, occupied_edges, 
                            current_locs, current_locs_to_agent, constrained_agents)
        if cspibt_worked is False and lacam_or_cspibt == "CS-PIBT":
            print("CS-PIBT ERROR!")
            raise RuntimeError('CS-PIBT failed for agent {}; should never fail without LaCAM constraints!', agent_id)
        if cspibt_worked is False:
            break

    return move_matrix, cspibt_worked
    


def main(args: argparse.ArgumentParser):
    # Load the map
    if not os.path.exists(args.mapNpzFile):
        raise FileNotFoundError('Map file: {} not found.'.format(args.mapNpzFile))
    map_npz = np.load(args.mapNpzFile) # Keys are {MAPNAME}.map -> (H,W)
    if args.mapName+".map" not in map_npz:
        raise ValueError('Map name not found in the map file.')
    map_grid = map_npz[args.mapName+".map"] # (H,W)

    # Load the scen
    if not os.path.exists(args.scenFile):
        raise FileNotFoundError('Scen file: {} not found.'.format(args.scenFile))
    start_locations, goal_locations = parse_scene(args.scenFile) # Each (max agents,2)
    num_agents = args.agentNum # This is N
    if start_locations.shape[0] < num_agents:
        raise ValueError('Not enough agents in the scen file.')
    start_locations = start_locations[:num_agents] # (N,2)
    goal_locations = goal_locations[:num_agents] # (N,2)

    # Load the bd
    # pdb.set_trace()
    if not os.path.exists(args.bdNpzFile):
        raise FileNotFoundError('BD file: {} not found.'.format(args.bdNpzFile))
    scen_num = args.scenFile.split('-')[-1].split('.')[0]
    bd_key = f"{args.mapName}-random-{scen_num}{num_agents}"
    bd_npz = np.load(args.bdNpzFile)
    if bd_key not in bd_npz:
        raise ValueError('BD key {} not found in the bd file'.format(bd_key))
    bd = bd_npz[bd_key] # (max agents,H,W)

    # Load the model
    k = args.k
    if not os.path.exists(args.modelPath):
        raise FileNotFoundError('Model file: {} not found.'.format(args.modelPath))
    model = torch.load(args.modelPath)
    model.eval()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Simulate
    solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success = simulate(model, 
            k, args.m, map_grid, bd, start_locations, goal_locations, 
            args.maxSteps, args.shieldType, args.seed)
    
    # Save the statistics into the csv file
    # if os.path.exists(args.outputCSVFile):
    #     df = pd.read_csv(args.outputCSVFile)
    # else:
    #     df = pd.DataFrame(columns=['map', 'scen', 'agentNum', 'seed', 'shieldType', 
    #                                'success', 'total_cost_true', 'total_cost_not_resting_at_goal',
    #                                'num_agents_at_goal'])
    # new_df = pd.DataFrame.from_dict({"map": [args.mapName], 'scen': [args.scenFile], 'agentNum': [args.agentNum],
    #                                 'seed': [args.seed], 'shieldType': [args.shieldType], 'success': [success],
    #                                 'total_cost_true': [total_cost_true], 'total_cost_not_resting_at_goal': [total_cost_not_resting_at_goal],
    #                                 'num_agents_at_goal': [num_agents_at_goal]})
    # df = pd.concat([df, new_df], ignore_index=True)
    # df.to_csv(args.outputCSVFile, index=False)

    if not os.path.exists(args.outputCSVFile):
        with open(args.outputCSVFile, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['map', 'scen', 'agentNum', 'seed', 'shieldType',
                             'modelPath', 'k', 'm', 'maxSteps', 
                             'success', 'total_cost_true', 'total_cost_not_resting_at_goal',
                             'num_agents_at_goal'])
            
    with open(args.outputCSVFile, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([args.mapName, args.scenFile, args.agentNum, args.seed, args.shieldType, 
                         args.modelPath, args.k, args.m, args.maxSteps,
                         success, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal])

    # Save the paths
    assert(args.outputPathsFile.endswith('.npy'))
    np.save(args.outputPathsFile, solution_path)

### Example run
# python -m gnn.simulator2 --mapNpzFile data_collection/data/benchmark_data/constant_npzs/all_maps.npz
#       --mapName den520d --scenFile data_collection/data/benchmark_data/scens/den520d-random-1.scen
#       --agentNum 100 --bdNpzFile data_collection/data/benchmark_data/constant_npzs/den520d_bds.npz
#       --modelPath data_collection/data/logs/EXP_Test2/iter0/models/max_test_acc.pt 
#       --k=4 --m=5
#       --maxSteps 200 --seed 0 --shieldType CS-PIBT
#       --outputCSVFile data_collection/data/logs/EXP_Test2/iter0/results.csv
#       --outputPathsFile data_collection/data/logs/EXP_Test2/iter0/paths.npy
if __name__ == '__main__':
    # testGetCosts()
    parser = argparse.ArgumentParser()
    # Map / scen parameters
    parser.add_argument('--mapNpzFile', type=str, required=True)
    parser.add_argument('--mapName', type=str, help="Without .map", required=True)
    parser.add_argument('--scenFile', type=str, required=True)
    parser.add_argument('--agentNum', type=int, required=True)
    parser.add_argument('--bdNpzFile', type=str, required=True)
    # Simulator parameters
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--k', type=int, help="local window size", required=True)
    parser.add_argument('--m', type=int, help="number of closest neighbors", required=True)
    parser.add_argument('--maxSteps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shieldType', type=str, default='CS-PIBT')
    # Output parameters
    parser.add_argument('--outputCSVFile', type=str, help="where to output statistics", required=True)
    parser.add_argument('--outputPathsFile', type=str, help="where to output path", required=True)
    args = parser.parse_args()

    if args.mapName.endswith('.map'): # Remove ending .map
        args.mapName = args.mapName.removesuffix('.map')
    
    main(args)