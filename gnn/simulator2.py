import os
import argparse
import pdb
import numpy as np
import torch # For the model
import pandas as pd # For saving the results
import csv # For saving the results
from collections import deque # For the queue of constraints

from gnn.dataloader import create_data_object, normalize_graph_data
from gnn.trainer import GNNStack, CustomConv # Required for using the model even if not explictly called
from custom_utils.common_helper import str2bool, getMapBDScenAgents

####################################################
#### Helper functions
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
    total_cost_not_resting_at_goal = (1-resting_at_goal).sum() # int

    num_agents_at_goal = np.sum(at_goal[-1]) # int
    assert(total_cost_true >= total_cost_not_resting_at_goal)
    assert(total_cost_true <= (solution_path.shape[0]-1)*solution_path.shape[1])
    return total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal

def testGetCosts():
    goal_locs = np.array([[1,10], [2,20], [3,30]]) # (N=3,2)
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

####################################################
#### LaCAM and PIBT func

LABEL_TO_MOVES = np.array([[0,0], [0,1], [1,0], [-1,0], [0,-1]]) #  Stop, Right, Down, Up, Left
 # This needs to match Pipeline's action ordering

def pibtRecursive(grid_map, agent_id, action_preferences, planned_agents, move_matrix, 
         occupied_nodes, occupied_edges, current_locs, current_locs_to_agent,
         constrained_agents_to_action):
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
        constrained_agents_to_action: dict: agent_id -> action_index
    """
    def findAgentAtLocation(aLoc):
        if tuple(aLoc) in current_locs_to_agent.keys():
            return current_locs_to_agent[tuple(aLoc)]
        else:
            return -1

    moves_ordered = LABEL_TO_MOVES[action_preferences[agent_id]]
    if agent_id in constrained_agents_to_action.keys(): # Force agent to only pick that action if constrained
        action_index = constrained_agents_to_action[agent_id]
        moves_ordered = moves_ordered[action_index:action_index+1] # Only consider that action
        print("Agent {} constrained to action {}".format(agent_id, action_index))

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
            isvalid = pibtRecursive(grid_map, conflicting_agent, action_preferences, planned_agents,
                                move_matrix, occupied_nodes, occupied_edges, current_locs,
                                current_locs_to_agent, constrained_agents_to_action)
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

def pibt(grid_map, action_preferences, current_locs, agent_priorities, agent_constraints):
    """Inputs:
        grid_map: (H,W)
        action_preferences: (N,5)
        current_locs: (N,2)
        agent_priorities: (N)
        agent_constraints: [(agent_id, action index), ...], empty list if no constraints
    """
    agent_order = np.argsort(-agent_priorities) # Sort by priority, highest first
    move_matrix = np.zeros((len(agent_priorities), 2), dtype=int) # (N,2)
    occupied_nodes = []
    occupied_edges = []
    planned_agents = []

    current_locs_to_agent = dict() # Maps (row, col) -> agent_id
    for agent_id in agent_order:
        if tuple(current_locs[agent_id]) not in current_locs_to_agent.keys():
            current_locs_to_agent[tuple(current_locs[agent_id])] = agent_id
        else:
            raise RuntimeError('Multiple agents at same location!')

    ### Convert agent_id constraints to actual agent indices based on agent_order
    # This means that constraints apply to highest priority agents first, which is desired as these agents are furthest from their goals
    constrained_agents_to_action = dict()
    for agent_id, action_index in agent_constraints:
        which_agent = agent_order[agent_id]
        constrained_agents_to_action[which_agent] = action_index

    ### Plan agents in order of priority
    for agent_id in agent_order:
        if agent_id in planned_agents:
            continue
        pibt_worked = pibtRecursive(grid_map, agent_id, action_preferences, planned_agents, 
                            move_matrix, occupied_nodes, occupied_edges, 
                            current_locs, current_locs_to_agent, constrained_agents_to_action)
        if pibt_worked is False:
            break

    return move_matrix, pibt_worked


def lacam(start_locations, goal_locations, bd, grid_map, conversion_type, 
                            k, m, model, device):
    """Inputs:
        bd: (N,H,W)
        grid_map: (H,W)
    """

    class HLNode:
        def __init__(self, state, action_preferences, parent) -> None:
            self.state = state
            self.action_preferences = action_preferences
            self.parent = parent
            self.queue_of_constraints = deque() # Each element is a list of tuples (agentId, actionIndex)
            # A successor could require constraining multiple agents, therefore each constraint is a list of tuples (one per constrained agent)
            # Conceptually, we are lazily BFSing the constraints. When generating [(0,3),(1,4)] 
            # we want to add [(0,3),(1,4),(2,0)], [(0,3),(1,4),(2,1)], ... [(0,3),(1,4),(2,4)]
            self.queue_of_constraints.append([]) # Start with no constraints

            ### Compute agent priorities via PIBT rule
            distance_to_goal = bd[range(len(state)), state[:,0], state[:,1]] # (N)
            if parent is None:
                self.depth = 0
                self.agent_priorities = distance_to_goal / distance_to_goal.max() # Normalize to [0,1]
            else:
                self.depth = parent.depth + 1
                self.agent_priorities = self.parent.agent_priorities + 1
                self.agent_priorities[distance_to_goal == 0] = 0 # Set priority to 0 if reached goal


        def getNextState(self):
            """Outputs:
                new_state: None or (N,2)
            """
            if len(self.queue_of_constraints) == 0:
                return False, None, None, True
            curConstraint = self.queue_of_constraints.popleft()
            # curConstraint is a list of tuples (agentId, actionIndex), agentId of K correponds to agent with K highest priority
            
            ### Add in next constraints
            if len(curConstraint) == 0: # Initial, start with constraining agent 0
                for i in range(0,5):
                    self.queue_of_constraints.append([(0,i)])
            else:
                # curConstraint is a list of tuples (agentId, actionIndex) dictating that agentId should take actionIndex
                curAgent = curConstraint[-1][0] # curConstraint[-1] is the last constraint added
                for i in range(0,5): # We want to constrain the next agent with the same current constraints + the new constraint
                    self.queue_of_constraints.append(curConstraint + [(curAgent+1,i)])

            # Run PIBT
            new_move, pibt_worked = pibt(grid_map, self.action_preferences, self.state, self.agent_priorities, curConstraint)

            if not pibt_worked: # Failed with the constraints
                return None
            new_state = self.state + new_move
            return new_state


    mainStack = deque() # Stack of HLNodes
    stateToHLNodes = dict() # Maps str(state) to HLNode

    ### Initialize the HLNode
    curNode = HLNode(start_locations, None)
    mainStack.appendleft(curNode) # Start with the initial state
    stateToHLNodes[str(start_locations)] = curNode

    success = False
    MAXNODECOUNT = 1000
    totalNodesExpanded = 0
    while len(mainStack) > 0:
        curNode : HLNode = mainStack.popleft()
        if len(curNode.queue_of_constraints) != 0: # Always add in the original HLNode if not exhausted
            mainStack.appendleft(curNode)
        new_locs = curNode.getNextState()
        if new_locs is None:
            continue
        totalNodesExpanded += 1

        # Check if all agents have reached their goals
        if np.all(np.equal(new_locs, goal_locations)):
            break

        if str(new_locs) in stateToHLNodes.keys(): # Already visited/created this state
            curNode = stateToHLNodes[str(new_locs)] # Get the existing HLNode
        else:
            # Create a new HLNode
            probs = runNNOnState(new_locs, bd, grid_map, k, m, model, device)
            action_preferences = convertProbsToPreferences(probs, conversion_type) # (N,5)
            newHLNode = HLNode(new_locs, action_preferences, curNode)
            mainStack.appendleft(newHLNode)

    # Get path via backtracking. If not a success, this returns the path to the last state found
    entirePath = [new_locs]
    while curNode is not None:
        entirePath.append(curNode.state)
        curNode = curNode.parent
    entirePath.reverse() # Reverse to get path from start to goal

def lacamOrPibt(lacam_or_cspibt, grid_map, action_preferences, current_locs, 
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
            # print("UH OH, MULTIPLE AGENTS AT SAME LOCATION!")
            # pdb.set_trace()
            raise RuntimeError('Multiple agents at same location!')

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
        cspibt_worked = pibtRecursive(grid_map, agent_id, action_preferences, planned_agents, 
                            move_matrix, occupied_nodes, occupied_edges, 
                            current_locs, current_locs_to_agent, constrained_agents)
        if cspibt_worked is False and lacam_or_cspibt == "CS-PIBT":
            print("CS-PIBT ERROR!")
            raise RuntimeError('CS-PIBT failed for agent {}; should never fail without LaCAM constraints!', agent_id)
        if cspibt_worked is False:
            break

    return move_matrix, cspibt_worked
    

def runNNOnState(cur_locs, bd, grid_map, k, m, model, device):
    """Inputs:
        cur_locs: (N,2)
    Outputs:
        probs: (N,5)
    """

    with torch.no_grad():
        # Create the data object
        data = create_data_object(cur_locs, bd, grid_map, k, m)
        data = normalize_graph_data(data, k)
        data = data.to(device)

        # Forward pass
        _, predictions = model(data)
        probabilities = torch.softmax(predictions, dim=1) # More general version

        # Get the action preferences
        probs = probabilities.cpu().detach().numpy() # (N,5)

    return probs


def simulate(device, model, k, m, grid_map, bd, start_locations, goal_locations, 
             max_steps, shield_type):
    """Inputs:
        grid_map: (H,W), note includes padding
        bd: (N,H,W), note includes padding
        start_locations: (N,2)
        goal_locations: (N,2)
    """
    cur_locs = start_locations # (N,2)
    # Ensure no start/goal locations are on obstacles
    assert(grid_map[start_locations[:,0], start_locations[:,1]].sum() == 0)
    assert(grid_map[goal_locations[:,0], goal_locations[:,1]].sum() == 0)

    # agent_priorities = np.random.permutation(len(cur_locs)) # (N)
    agent_priorities = bd[range(len(start_locations)), start_locations[:,0], start_locations[:,1]] # (N)
    agent_priorities = agent_priorities / agent_priorities.max() # Normalize to [0,1]

    solution_path = [cur_locs.copy()]
    success = False
    for step in range(max_steps):
        # Update priorities
        agents_at_goal = np.all(np.equal(cur_locs, goal_locations), axis=1) # (N)
        agent_priorities[agents_at_goal] = 0 # Agents at goal have priority 0
        agent_priorities[agents_at_goal == False] += 1 # Agents not at goal increase priority by 1

        probs = runNNOnState(cur_locs, bd, grid_map, k, m, model, device)

        action_preferences = convertProbsToPreferences(probs, "sampled") # (N,5)

        # Run the shield
        new_move, cspibt_worked = lacamOrPibt(shield_type, grid_map, action_preferences, cur_locs, 
                                        agent_priorities, [])
        if not cspibt_worked:
            raise RuntimeError('CS-PIBT failed; should never fail when no using LaCAM constraints!')
        cur_locs = cur_locs + new_move # (N,2)
        # cur_locs += new_move # (N,2)
        solution_path.append(cur_locs.copy())
        assert(np.all(grid_map[cur_locs[:,0], cur_locs[:,1]] == 0)) # Ensure no agents are on obstacles

        # Check if all agents have reached their goals
        if np.all(np.equal(cur_locs, goal_locations)):
            success = True
            break
    
    # pdb.set_trace()
    solution_path = np.array(solution_path) # (T<=max_steps+1,N,2)
    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal = getCosts(solution_path, goal_locations)

    return solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success


def main(args: argparse.ArgumentParser):
    # Setting constants
    torch.set_num_threads(1) # Make pytorch use only 1 thread, otherwise by default will try using all threads    
    k = args.k

    # Load the map
    if not os.path.exists(args.mapNpzFile):
        raise FileNotFoundError('Map file: {} not found.'.format(args.mapNpzFile))
    map_npz = np.load(args.mapNpzFile) # Keys are {MAPNAME}.map -> (H,W)
    if args.mapName+".map" not in map_npz:
        raise ValueError('Map name not found in the map file.')
    map_grid = map_npz[args.mapName+".map"] # (H,W)
    map_grid = np.pad(map_grid, k, 'constant', constant_values=1) # Add padding

    # Load the scen
    if not os.path.exists(args.scenFile):
        raise FileNotFoundError('Scen file: {} not found.'.format(args.scenFile))
    start_locations, goal_locations = parse_scene(args.scenFile) # Each (max agents,2)
    num_agents = args.agentNum # This is N
    if start_locations.shape[0] < num_agents:
        raise ValueError('Not enough agents in the scen file.')
    start_locations = start_locations[:num_agents] + k # (N,2)
    goal_locations = goal_locations[:num_agents] + k # (N,2)

    # Load the bd
    # pdb.set_trace()
    if not os.path.exists(args.bdNpzFile):
        raise FileNotFoundError('BD file: {} not found.'.format(args.bdNpzFile))
    scen_num = args.scenFile.split('-')[-1].split('.')[0]
    bd_key = f"{args.mapName}-random-{scen_num}"
    bd_npz = np.load(args.bdNpzFile)
    if bd_key not in bd_npz:
        raise ValueError('BD key {} not found in the bd file'.format(bd_key))
    bd = bd_npz[bd_key][:num_agents] # (max agents,H,W)->(N,H,W)
    bd = np.pad(bd, ((0,0),(k,k),(k,k)), 'constant', constant_values=12345678) # Add padding

    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.useGPU else "cpu") # Use GPU if available
    if not os.path.exists(args.modelPath):
        raise FileNotFoundError('Model file: {} not found.'.format(args.modelPath))
    model = torch.load(args.modelPath, map_location=device)
    model.eval()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get max steps
    if args.maxSteps.endswith('x'):
        longest_single_path = bd[range(num_agents), start_locations[:,0], start_locations[:,1]].max()
        max_steps = int(args.maxSteps[:-1]) * longest_single_path
    else:
        max_steps = int(args.maxSteps)

    # Simulate
    solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success = simulate(device,
            model, k, args.m, map_grid, bd, start_locations, goal_locations, 
            max_steps, args.shieldType)
    solution_path = solution_path - k # (T,N,2) Removes padding
    goal_locations = goal_locations - k # (N,2) Removes padding
    
    # Save the statistics into the csv file
    if not os.path.exists(args.outputCSVFile):
        # Create the file and write the header
        with open(args.outputCSVFile, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['mapName', 'scenFile', 'agentNum', 'seed', 'shieldType',
                             'modelPath', 'useGPU', 'k', 'm', 'maxSteps', 
                             'success', 'total_cost_true', 'total_cost_not_resting_at_goal',
                             'num_agents_at_goal'])
            
    with open(args.outputCSVFile, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([args.mapName, args.scenFile, args.agentNum, args.seed, args.shieldType, 
                         args.modelPath, args.useGPU, args.k, args.m, args.maxSteps,
                         success, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal])

    # Save the paths
    assert(args.outputPathsFile.endswith('.npy'))
    np.save(args.outputPathsFile, solution_path)

    # Create the scen files
    numToCreate = args.numScensToCreate
    # Always include the first timestep + last 5 timesteps, then sample the rest
    sampled_timesteps = np.random.choice(solution_path.shape[0], numToCreate-6, replace=False)
    sampled_timesteps = np.concatenate([[0], sampled_timesteps, np.arange(solution_path.shape[0]-5, solution_path.shape[0])])
    # sampled_timesteps = np.random.choice(solution_path.shape[0], numToCreate, replace=False)
    for t in sampled_timesteps:
        # scenFilepath = args.outputScenPrefix + f".{t}.scen"
        mapname, bdname, scenname, _ = getMapBDScenAgents(args.scenFile)
        custom_scenname = f"{scenname}_t{t}"
        prefix = os.path.dirname(args.outputPathsFile)
        scenFilepath = f"{prefix}/{bdname}.{custom_scenname}.{num_agents}.scen"
        # pdb.set_trace()
        createScenFile(solution_path[t], goal_locations, args.mapName, scenFilepath)

### Example command
"""
python -m gnn.simulator2 --mapNpzFile data_collection/data/benchmark_data/constant_npzs/all_maps.npz \
      --mapName random_32_32_10 --scenFile data_collection/data/benchmark_data/scens/random_32_32_10-random-1.scen \
      --agentNum=10 --bdNpzFile data_collection/data/benchmark_data/constant_npzs/random_32_32_10_bds.npz \
      --modelPath=data_collection/data/logs/EXP_Small/iter29/models/max_test_acc.pt --useGPU=False \
      --k=4 --m=5 \
      --maxSteps=200 --seed 0 --shieldType CS-PIBT \
      --outputCSVFile data_collection/data/logs/EXP_Test4/iter0/results.csv \
      --outputPathsFile data_collection/data/logs/EXP_Test4/iter0/encountered_scens/paths.npy \
      --numScensToCreate 10 --outputScenPrefix data_collection/data/logs/EXP_Test4/iter0/encountered_scens/den520d/den520d-random-1.scen100
"""
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
    parser.add_argument('--useGPU', type=lambda x: bool(str2bool(x)), required=True)
    parser.add_argument('--k', type=int, help="local window size", required=True)
    parser.add_argument('--m', type=int, help="number of closest neighbors", required=True)
    parser.add_argument('--maxSteps', type=str, help="int or [int]x, e.g. 100 or 2x to denote multiplicative factor", required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shieldType', type=str, default='CS-PIBT')
    # Output parameters
    parser.add_argument('--outputCSVFile', type=str, help="where to output statistics", required=True)
    parser.add_argument('--outputPathsFile', type=str, help="where to output path, ends with .npy", required=True)
    parser.add_argument('--numScensToCreate', type=int, help="how many scens to create", required=True)
    parser.add_argument('--outputScenPrefix', type=str, help="output prefix to create scens", required=False)

    args = parser.parse_args()

    if args.mapName.endswith('.map'): # Remove ending .map
        args.mapName = args.mapName.removesuffix('.map')
    if args.outputScenPrefix is None:
        tmp = args.outputPathsFile.removesuffix('.npy')
    
    main(args)