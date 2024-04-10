'''
New Agent Simulator Environment
Utilizes matrix for speeding up the performance of the simulator
Built upon the fundamental vesion of Qingbiao's simulator
-------------------------
Ver. 1.0 beta
Weizhe Lin @ 12/05/2020
Ver. 2.0
Weizhe Lin @ 11/08/2020
-------------------------
'''

import torch

import os
import sys
import yaml
import numpy as np
import random
import time
import scipy.io as sio
import shutil
import pdb

random.seed(1337)

from torch import nn
import utils.graphUtils.graphTools as graph
from scipy.spatial.distance import squareform, pdist
# from dataloader.statetransformer import AgentState
# from dataloader.statetransformer_localGuidance import AgentState
# from dataloader.statetransformer_localGuidance_SDObs import AgentState
# from dataloader.statetransformer_localGuidance_SemiLocal import AgentState

# from dataloader.statetransformer_globalGuidance import AgentState
# from dataloader.statetransformer_Guidance import AgentState


# from onlineExpert.ECBS_onlineExpert import ComputeCBSSolution


class multiRobotSimNew:

    def __init__(self):
        '''
        Simulator init method
        Args:
            config: global config for simulation
        '''
        # print('*****New Simulator Enabled*****')
        # self.config = config

        # # Init AgentState
        # # self.AgentState = AgentState(self.config)

        # # Per-define directions
        self.up = np.array([-1, 0])
        self.down = np.array([1, 0])
        self.left = np.array([0, -1])
        self.right = np.array([0, 1])
        self.stop = np.array([0, 0])
        self.up_keyValue = 0
        self.down_keyValue = 2
        self.left_keyValue = 1
        self.right_keyValue = 3
        self.stop_keyValue = 4

        # self.current_positions = None
        # self.start_positions = None
        # self.goal_positions = None
        # self.obstacle_positions = None
        # self.num_obstacle = None
        # self.reach_goal = None
        # self.first_move = None

        # self.List_MultiAgent_ActionVec_target = None
        # self.store_MultiAgent = None
        # self.channel_map = None

        # self.size_map = None
        # self.maxstep = None

        # self.posObstacle = None
        # self.numObstacle = None
        # self.posStart = None
        # self.posGoal = None

        # self.currentState_predict = None

        # self.makespanTarget = None
        # self.flowtimeTarget = None
        # self.makespanPredict = None
        # self.flowtimePredict = None

        # self.count_reachgoal = None
        # self.count_reachgoalTarget = None
        # self.fun_Softmax = None

        # self.zeroTolerance = 1e-9

        # # Link function self.getAdjacencyMatrix
        # if self.config.dynamic_commR:
        #     # comm radius that ensure initial graph connected
        #     print("run on multirobotsim (radius dynamic) with collision shielding")
        #     self.getAdjacencyMatrix = self.computeAdjacencyMatrix
        # else:
        #     # comm radius fixed
        #     print("run on multirobotsim (radius fixed) with collision shielding")
        #     self.getAdjacencyMatrix = self.computeAdjacencyMatrix_fixedCommRadius

        # self.failureCases_input = self.config.failCases_dir + 'input/'
        # self.dir_sol = os.path.join(self.config.failCases_dir, "output_ECBS/")

        # self.shieldType = self.config.shieldType
        # self.pibt_r = self.config.pibt_r
        # self.shieldTime = 0
        # assert(self.shieldType in ["Default", "LaCAM"])

    # def setup(self, loadInput, loadTarget, case_config, tensor_map, ID_dataset, mode):
    #     '''
    #     Setup of environment, called before actual use.
    #     Args:
    #         loadInput: agent initial/goal positions
    #         loadTarget: computed expert paths of all agents
    #         case_config: contains information of a case (ID_MAP, ID_case, makespanTarget)
    #         tensor_map: information of the map including obstacles
    #         ID_dataset: dataset id
    #         mode: - test_trainingSet using softmax for action decoding
    #             - otherwise using exp_multinorm for action decoding

    #     Returns:

    #     '''
    #     # makespanTarget is total steps of expert solutions
    #     ID_MAP, ID_case, makespanTarget = case_config

    #     # Here defines the action decoding function used in this simulator
    #     if self.config.default_actionSelect:
    #         if mode == 'test_trainingSet':
    #             self.convectToActionKey = self.convectToActionKey_softmax
    #         else:
    #             self.convectToActionKey = self.convectToActionKey_exp_multinorm
    #     else:
    #         if self.config.action_select == 'soft_max':
    #             self.convectToActionKey = self.convectToActionKey_softmax

    #         elif self.config.action_select == 'sum_multinorm':
    #             self.convectToActionKey = self.convectToActionKey_sum_multinorm

    #         elif self.config.action_select == 'exp_multinorm':
    #             self.convectToActionKey = self.convectToActionKey_exp_multinorm

    #     self.ID_MAP = int(ID_MAP)
    #     self.ID_case = int(ID_case)
    #     # self.fun_Softmax = nn.Softmax(dim=-1)
    #     self.fun_Softmax = nn.LogSoftmax(dim=-1)
    #     self.ID_dataset = ID_dataset

    #     self.store_GSO = []
    #     self.store_attentionGSO = []
    #     self.store_localPath = []
    #     self.store_communication_radius = []
    #     self.status_MultiAgent = {}
    #     target = loadTarget.permute(1, 2, 3, 0)
    #     self.List_MultiAgent_ActionVec_target = target[:, :, :, 0]
    #     # self.List_MultiAgent_ActionVec_target = target[:, :, 0]

    #     # Read map
    #     self.channel_map = tensor_map[0]  # setupState[:, :, 0, 0, 0]
    #     # self.AgentState.setmap(self.channel_map)

    #     # Read obstacles
    #     self.obstacle_positions = self.get_pos(self.channel_map)
    #     self.num_obstacle = self.obstacle_positions.shape[0]

    #     # Save wall info into a dict
    #     self.wall_dict = {}
    #     for i in range(self.num_obstacle):
    #         tuple_pos = tuple(self.obstacle_positions[i])
    #         self.wall_dict[tuple_pos] = i

    #     # print("obstacle_positions shape: ", self.obstacle_positions.shape)
    #     self.size_map = self.channel_map.shape

    #     # Read and init agent positions
    #     self.start_positions = loadInput[0, 1, :, :].cpu().numpy()
    #     self.current_positions = self.start_positions.copy()
    #     # Read goal positions
    #     self.goal_positions = loadInput[0, 0, :, :].cpu().numpy()
    #     self.agent_priorities = np.sum(np.abs(self.current_positions - self.goal_positions), axis=-1) # RVMod (N,2)->(N)
    #     # save paths during the simulation
    #     self.path_list = []
    #     self.path_list.append(self.current_positions.copy())
    #     self.path_matrix = None

    #     self.BDs = np.load('BDs/random-32-32-10-scen-{}.npy'.format(int(ID_case.numpy())))

    #     # print('position vector:', self.start_positions.shape, self.goal_positions.shape)

    #     self.reach_goal = np.zeros(self.config.num_agents, )
    #     self.first_move = np.zeros(self.config.num_agents, ) + 1 # RVMod
    #     self.end_step = np.zeros(self.config.num_agents, )

    #     self.expert_first_move = np.zeros(self.config.num_agents, )
    #     self.expert_end_step = np.zeros(self.config.num_agents, )
    #     self.expert_path_list = []
    #     self.expert_path_matrix = None

    #     self.posObstacle = self.findpos(self.channel_map).to(self.config.device)
    #     self.numObstacle = self.posObstacle.shape[0]
    #     self.size_map = self.channel_map.shape

    #     if self.config.num_agents >= 20:
    #         self.rate_maxstep = 3
    #     else:
    #         self.rate_maxstep = self.config.rate_maxstep

    #     # Calculate maximum allowed steps in this simulation, otherwise timeout
    #     self.maxstep = int(makespanTarget.type(torch.int32) * self.rate_maxstep)

    #     # self.check_predictCollsion = False
    #     self.check_moveCollision = True
    #     self.check_predictEdgeCollsion = [False] * self.config.num_agents
    #     self.count_reachgoal = [False] * self.config.num_agents
    #     self.count_reachgoalTarget = [False] * self.config.num_agents
    #     self.allReachGoal_Target = False
    #     self.makespanTarget = 0
    #     self.flowtimeTarget = 0
    #     self.makespanPredict = self.maxstep
    #     self.flowtimePredict = self.maxstep * self.config.num_agents  # 0

    #     # Process expert solution for metrics comparison
    #     self.getPathTarget()

    # def getPathTarget(self):
    #     '''
    #     Process the expert solution.
    #     Read the information including flowtime, makespan and all steps from the expert solution
    #     # todo check the length for ground truth, out of index
    #     # TODO: move this function to data transformer
    #     Returns:

    #     '''

    #     current_pos = self.start_positions.copy()
    #     self.expert_path_list.append(current_pos.copy())
    #     for current_step in range(self.List_MultiAgent_ActionVec_target.shape[1]):
    #         # Following each step in the solution to combine information
    #         # Lin: updated to use matrix operations for speedup
    #         expert_output = self.List_MultiAgent_ActionVec_target[:, current_step, :].cpu().numpy()
    #         expert_actions = np.argmax(expert_output, axis=1)
    #         # print(expert_actions)
    #         expert_moves = np.zeros((self.config.num_agents, 2))

    #         expert_moves[expert_actions == self.up_keyValue, :] = self.up
    #         expert_moves[expert_actions == self.left_keyValue, :] = self.left
    #         expert_moves[expert_actions == self.down_keyValue, :] = self.down
    #         expert_moves[expert_actions == self.right_keyValue, :] = self.right
    #         expert_moves[expert_actions == self.stop_keyValue, :] = self.stop

    #         # update first step to move for each agent
    #         self.expert_first_move[
    #             (expert_actions != self.stop_keyValue) & (self.expert_first_move == 0)] = current_step + 1

    #         # update expert moves
    #         current_pos += expert_moves

    #         current_distance = np.sum(np.abs(current_pos - self.goal_positions), axis=1)
    #         # print(current_distance)
    #         # Update end_step

    #         self.expert_end_step[(current_distance == 0) & (self.expert_end_step == 0)] = current_step + 1

    #         # save current step to path matrix
    #         self.expert_path_list.append(current_pos.copy())

    #     self.expert_path_matrix = np.array(self.expert_path_list)

    #     # expert action length
    #     self.expert_action_length = self.expert_end_step - self.expert_first_move + 1

    #     # calculate flowtime
    #     self.flowtimeTarget = np.sum(self.expert_action_length)

    #     # maximum makespan
    #     self.makespanTarget = np.max(self.expert_end_step) - np.min(self.expert_first_move) + 1

    # def getCurrentState(self, return_GPos=False):

    #     store_goalAgents = torch.FloatTensor(self.goal_positions)
    #     store_stateAgents = torch.FloatTensor(self.current_positions)

    #     # tensor_currentState = self.AgentState.toInputTensor(store_goalAgents, store_stateAgents)
    #     tensor_currentState = tensor_currentState.unsqueeze(0)

    #     # agents_localPath = self.AgentState.get_localPath()
    #     # self.store_localPath.append(agents_localPath)

    #     if return_GPos:
    #         return tensor_currentState, store_stateAgents.unsqueeze(0)
    #     else:
    #         return tensor_currentState

    # def getGSO(self, step):
    #     '''
    #     Compute GSO (adjacency matrix) of the current agent graph,
    #     physically truncated according to radius of communication.
    #     Args:
    #         step:

    #     Returns:
    #         GSO_tensor: the tensor of the current GSO
    #     '''
    #     store_PosAgents = self.current_positions[None, :]  # Add new axis to fit input

    #     if step == 0:
    #         self.initCommunicationRadius()
    #     # print("{} - Step-{} - initCommunication Radius:{}".format(self.ID_dataset, step, self.communicationRadius))

    #     # comm radius fixed
    #     # GSO, communicationRadius, graphConnected = self.computeAdjacencyMatrix_fixedCommRadius(step, store_PosAgents, self.communicationRadius)

    #     # comm radius that ensure initial graph connected
    #     GSO, communicationRadius, graphConnected = self.getAdjacencyMatrix(step, store_PosAgents,
    #                                                                        self.communicationRadius)
    #     GSO_tensor = torch.from_numpy(GSO)

    #     self.store_GSO.append(GSO)
    #     self.store_communication_radius.append(communicationRadius)
    #     return GSO_tensor

    # def record_attentionGSO(self, attentionGSO):
    #     '''
    #     This function helps the model to save output attention graphs for visualisation.
    #     Args:
    #         attentionGSO: tensor of GSO after the GAT attention module (values 0~1)

    #     Returns:

    #     '''
    #     self.store_attentionGSO.append(attentionGSO)

    # RVMod
    def lacam_check_collision(self, actionPreds, current_positions, agent_priorities, num_agents, size_map, map):
        '''
        Runs LaCAM and PIBT instead of naive collision checking
        Args:
            actionPreds: (N,5) action preds
        Returns:
            new_move: valid move without collision
            out_boundary: matrix of whether the agent in place goes out of boundary
            move_to_wall: ids of moving to walls
            collide_agents: ids of collided agents (earlier mover has advantages to move)
            collide_in_move_agents: ids of face-to-face collision (both stop)
        '''

        # agent_order = np.arange(self.config.num_agents)
        ### Recompute priorities
        # RVMod: Update self.agent_priorities if reached goal
        # current_distance = np.sum(np.abs(current_positions - self.goal_positions), axis=1)
        # self.agent_priorities = np.maximum(self.agent_priorities, current_distance) # Increase priority if further from goal
        # self.agent_priorities[current_distance == 0] = 0 # Set priority to 0 if reached goal
        agent_order = np.argsort(-agent_priorities) # Sort by priority, highest first
        moveMatrix = np.zeros((num_agents, 2))
        occupiedNodes = []
        occupiedEdges = []
        plannedAgents = []

        curLocations = set()
        for agent_id in agent_order:
            if tuple(current_positions[agent_id]) not in curLocations:
                curLocations.add(tuple(current_positions[agent_id]))
            else:
                print("UH OH, MULTIPLE AGENTS AT SAME LOCATION!")
                pdb.set_trace()

        for agent_id in agent_order:
            if agent_id in plannedAgents:
                continue
            # tmp1 = len(plannedAgents)
            current_pos = current_positions[agent_id]
            if tuple(current_pos) in occupiedNodes:
                pdb.set_trace()
            lacamWorked = self.pibt(agent_id, actionPreds, plannedAgents, moveMatrix, occupiedNodes, occupiedEdges, current_positions, num_agents, size_map)
            # if len(plannedAgents) > tmp1+1:
                # print("PIBT added {} agents to plannedAgents".format(len(plannedAgents)-tmp1))
            if lacamWorked is False:
                print("PIBT ERROR!")
                raise RuntimeError('PIBT failed for agent {}. Single-step PIBT should never fail!', agent_id)
                # lacamWorked = self.pibt(agent_id, actionPreferences, plannedAgents, moveMatrix, occupiedNodes, occupiedEdges)
                ### Fall back to regular collision checking???
        out_boundary = np.zeros(num_agents, dtype=bool)
        move_to_wall = []
        collide_agents = []
        collide_in_move_agents = []
        #  move, out_boundary == True, move_to_wall, collide_agents, collide_in_move_agents
        return moveMatrix, out_boundary, move_to_wall, collide_agents, collide_in_move_agents

    def pibt(self, agent_id, actionPreds, plannedAgents, moveMatrix, occupiedNodes, occupiedEdges, current_positions, num_agents, size_map, curmap):
        '''
        Runs PIBT for a single agent
        Args:
            agent_id: agent id
            move_preferences: (N,5) up-down-left-right-wait
        Returns:
            isvalid: whether we can find a feasible solution
        '''

        def findAgentAtLocation(aLoc):
            for a in range(num_agents):
                if a == agent_id:
                    continue
                if tuple(current_positions[a]) == tuple(aLoc):
                    return a
            return -1

        # Below matches __init__ action order
        moves_ordered = np.array([self.up, self.left, self.down, self.right, self.stop])
        curActionPreds = actionPreds[agent_id]
        # actionPreferences = (-curActionPreds).argsort() # If strict ordering
        logits = np.exp(curActionPreds)
        logits = logits / logits.sum()

        if self.pibt_r > 0: ### Incorporate BDs
            BD_scores = []
            for drow, dcol in moves_ordered:
                neighbor_row = int(current_positions[agent_id][0] + drow)
                neighbor_col = int(current_positions[agent_id][1] + dcol)
                if neighbor_row < 0 or neighbor_col < 0 or neighbor_row >= size_map[0] or neighbor_col >= size_map[1]:
                    BD_scores.append(9999)
                    continue
                neighbor_score = self.BDs[agent_id, neighbor_row, neighbor_col]
                if neighbor_score == np.inf:
                    BD_scores.append(9999)
                    continue
                BD_scores.append(neighbor_score)
            # assert(self.BDs[agent_id, current_positions[agent_id][0], current_positions[agent_id][1]] < 1000
            assert(BD_scores[-1] < 1000)
            weighted_scores = BD_scores + self.pibt_r * (1 - logits) # TODO: tune this
            # weighted_scores = weighted_scores / weighted_scores.sum()
            # pdb.set_trace()
            actionPreferences = weighted_scores.argsort() # If strict ordering, sorts min to max
        else:
            ### Randomly sort using logits
            actionPreferences = np.random.choice(5, size=5, replace=False, p=logits)

            ### Below is for sanity checking that PIBT with BDs works as expected (it does)
            # BD_scores = []
            # for drow, dcol in moves_ordered:
            #     neighbor_row = int(current_positions[agent_id][0] + drow)
            #     neighbor_col = int(current_positions[agent_id][1] + dcol)
            #     if neighbor_row < 0 or neighbor_col < 0 or neighbor_row >= self.size_map[0] or neighbor_col >= self.size_map[1]:
            #         BD_scores.append(9999)
            #         continue
            #     neighbor_score = self.BDs[agent_id, neighbor_row, neighbor_col]
            #     if neighbor_score == np.inf:
            #         BD_scores.append(9999)
            #         continue
            #     BD_scores.append(neighbor_score + np.random.uniform(0, 1))
            # # assert(self.BDs[agent_id, current_positions[agent_id][0], current_positions[agent_id][1]] < 1000
            # assert(BD_scores[-1] < 1000)
            # # weighted_scores = BD_scores + self.pibt_r * (1 - logits) # TODO: tune this
            # # weighted_scores = weighted_scores / weighted_scores.sum()
            # # pdb.set_trace()
            # actionPreferences = np.argsort(BD_scores) # If strict ordering, sorts min to max
        moves_ordered = moves_ordered[actionPreferences]

        current_pos = current_positions[agent_id]
        for aMove in moves_ordered:
            next_loc = current_pos + aMove
            
            # Skip if would leave map bounds
            if next_loc[0] < 0 or next_loc[0] >= size_map[0] or next_loc[1] < 0 or next_loc[1] >= size_map[1]:
                continue
            # Skip if obstacle
            if curmap[next_loc[0], next_loc[1]]==1:
                continue
            # Skip if vertex occupied by higher agent
            if tuple(next_loc) in occupiedNodes:
                continue
            # Skip if reverse edge occupied by higher agent
            if tuple([*next_loc, *current_pos]) in occupiedEdges:
                continue
            
            ### Pretend we move there
            moveMatrix[agent_id] = aMove
            # pdb.set_trace()
            plannedAgents.append(agent_id)
            occupiedNodes.append(tuple(next_loc))
            occupiedEdges.append(tuple([*current_pos, *next_loc]))
            conflictingAgent = findAgentAtLocation(next_loc)
            if conflictingAgent != -1 and conflictingAgent not in plannedAgents:
                # Recurse
                isvalid = self.pibt(conflictingAgent, actionPreds, plannedAgents,
                                    moveMatrix, occupiedNodes, occupiedEdges, current_positions, num_agents, size_map, curmap)
                if isvalid:
                    return True
                else:
                    # pdb.set_trace()
                    del plannedAgents[-1]
                    del occupiedNodes[-1]
                    del occupiedEdges[-1]
                    continue
            else:
                # No conflict
                return True
            
        # No valid move found
        return False

    # def check_collision(self, current_pos, move):
    #     '''
    #     This function checks collisions, and disable illegal movements of agents if needed.
    #     Args:
    #         current_pos: current position
    #         move: move matrix (all agents' proposed moves)

    #     Returns:
    #         new_move: valid move without collision
    #         out_boundary: matrix of whether the agent in place goes out of boundary
    #         move_to_wall: ids of moving to walls
    #         collide_agents: ids of collided agents (earlier mover has advantages to move)
    #         collide_in_move_agents: ids of face-to-face collision (both stop)
    #     '''

    #     map_width = self.size_map[0]
    #     map_height = self.size_map[1]

    #     # Avoid Memory invasion
    #     move = move.copy()
    #     # print('map size:', map_width, map_height)
    #     new_pos = current_pos + move

    #     # Stop agents running out of the arena
    #     out_boundary = (new_pos[:, 0] >= map_width) | (new_pos[:, 1] >= map_height) | (new_pos[:, 0] < 0) | (
    #             new_pos[:, 1] < 0)
    #     move[out_boundary == True] = self.stop

    #     # print('out_of_boundaries:', out_boundary)

    #     # Add 0.5 step to detect face-to-face collision
    #     new_pos = current_pos + move / 2
    #     check_dict = {}
    #     collide_in_move_agents = []
    #     for i in range(self.config.num_agents):
    #         tuple_pos = tuple(new_pos[i])
    #         if tuple_pos in check_dict.keys():
    #             j = check_dict[tuple_pos]
    #             collide_in_move_agents += [i, j]
    #             move[i] = self.stop
    #             move[j] = self.stop
    #             # print('collide in move, stop', i, current_pos[i])
    #             # print('collide in move, stop', j, current_pos[j])
    #         check_dict[tuple_pos] = i

    #     #     collide_agents = list(set(collide_agents))
    #     # print('collide in move:', collide_in_move_agents)

    #     # Update new proposed move
    #     new_pos = current_pos + move

    #     need_reverse_list = []

    #     check_dict = {}
    #     collision_check_dict = {}
    #     collide_agents = []
    #     move_to_wall = []

    #     for i in range(self.config.num_agents):
    #         # Reverse Wall collision
    #         tuple_pos = tuple(new_pos[i])
    #         if tuple_pos in self.wall_dict.keys():
    #             # Collide into walls
    #             move_to_wall.append(i)
    #             move[i] = self.stop

    #             # Mark to reverse all the paths
    #             need_reverse_list.append(current_pos[i])

    #             new_pos = current_pos + move
    #             tuple_pos = tuple(new_pos[i])
    #             collision_check_dict.setdefault(tuple_pos, []).append(i)
    #         else:
    #             collision_check_dict.setdefault(tuple_pos, []).append(i)

    #     while len(collision_check_dict) > 0:
    #         tuple_pos, inplace_agent_list = collision_check_dict.popitem()
    #         # print(tuple_pos, inplace_agent_list)
    #         if len(inplace_agent_list) > 1:  # more than 1 agents show on the same pos
    #             # print(inplace_agent_list, 'happen to show in', tuple_pos)
    #             selected_agent = random.choice(inplace_agent_list)
    #             for i in inplace_agent_list:
    #                 if (move[i] == self.stop).all():
    #                     # print('agent', i, 'holds its position')
    #                     selected_agent = i
    #             # print('agent', selected_agent, 'is chosen to move')
    #             for i in [x for x in inplace_agent_list if x != selected_agent]:
    #                 # print('reverse', i)
    #                 move[i] = self.stop
    #                 # Mark to reverse all the paths
    #                 need_reverse_list.append(current_pos[i])
    #                 # collision_check_dict.setdefault(tuple_pos, []).append(i)
    #             collide_agents += inplace_agent_list
    #     # Record all moves of agents, easier to reverse when some of them crash with reversed agents
    #     reverse_dict = {}
    #     new_pos = current_pos + move
    #     for i in range(self.config.num_agents):
    #         reverse_dict.setdefault(tuple(new_pos[i]), []).append((i, tuple(current_pos[i])))

    #     # Reverse invalid path
    #     # print('####### START REVERSING ######')
    #     # if len(need_reverse_list) != 0:
    #     #     print(need_reverse_list)
    #     #     print(reverse_dict)
    #     while len(need_reverse_list) > 0:
    #         need_reverse_pos = need_reverse_list.pop(0)
    #         # print('need_reverse_pos', need_reverse_pos)
    #         need_reverse_pos_tuple = tuple(need_reverse_pos)
    #         if need_reverse_pos_tuple in reverse_dict.keys():
    #             to_do_list = reverse_dict[need_reverse_pos_tuple]
    #             # print('to_do_list', to_do_list)
    #             for to_do_item in to_do_list:
    #                 reverse_agent, next_need_reverse_pos_tuple = to_do_item[0], to_do_item[1]
    #                 if need_reverse_pos_tuple != next_need_reverse_pos_tuple:
    #                     need_reverse_list.append(np.array(next_need_reverse_pos_tuple))
    #                 move[reverse_agent] = self.stop
    #                 # print('reverse', reverse_agent)
    #     # print('####### FINISHED REVERSING ######')
    #     #     collide_agents = list(set(collide_agents))
    #     # print('move into walls after move:', move_to_wall)
    #     # print('collide after move:', collide_agents)

    #     # # Validate collision (for validation only, commented out in real runs)
    #     # new_pos = current_pos + move
    #     # valid_dict = {}
    #     # # pause = False
    #     # for i in range(self.config.num_agents):
    #     #     tuple_pos = tuple(new_pos[i])
    #     #     if tuple_pos in valid_dict.keys():
    #     #         j = valid_dict[tuple_pos]
    #     #         # print(i, j, 'collision', tuple_pos)
    #     #         # pause = True
    #     #     valid_dict[tuple_pos] = i
    #     # # if pause:
    #     # #     a = input('stop')
    #     return move, out_boundary == True, move_to_wall, collide_agents, collide_in_move_agents

    # def move(self, actionVec, currentstep, agent_priorities, current_positions, end_step):
    #     # print('current step', currentstep)
    #     # allReachGoal = (np.count_nonzero(self.reach_goal) == self.config.num_agents)
    #     # print('++++++++++step:', currentstep)
    #     # print('new robot:', self.reach_goal)
    #     # print('current_pos', self.current_positions, self.goal_positions)
    #     # print('first_move:', self.first_move)
    #     # print('end_step:', self.end_step)

    #     check_predictCollsion = False
    #     check_moveCollision = False

    #     # if (not allReachGoal) and (currentstep < self.maxstep):

    #     if True:
    #         proposed_actions = [int(self.convectToActionKey(actionVec[id_agent]).cpu())
    #                             for id_agent in range(self.config.num_agents)]
    #         proposed_actions = np.array(proposed_actions)
    #         # print('model_output', proposed_actions)
    #         proposed_moves = np.zeros((self.config.num_agents, 2))

    #         proposed_moves[proposed_actions == self.up_keyValue, :] = self.up
    #         proposed_moves[proposed_actions == self.left_keyValue, :] = self.left
    #         proposed_moves[proposed_actions == self.down_keyValue, :] = self.down
    #         proposed_moves[proposed_actions == self.right_keyValue, :] = self.right
    #         proposed_moves[proposed_actions == self.stop_keyValue, :] = self.stop

    #         # update first step to move for each agent
    #         # self.first_move[(proposed_actions != self.stop_keyValue) & (self.first_move == 0)] = currentstep
    #         # Check collisions, update valid moves for each agent
    #         if self.shieldType == "Default":
    #             time_start = time.time()
    #             new_move, move_to_boundary, move_to_wall_agents, collide_agents, collide_in_move_agents = self.check_collision(
    #                 current_positions, proposed_moves)
    #             time_end = time.time()
    #             self.shieldTime += time_end - time_start
    #         elif self.shieldType == "LaCAM": ### RVMod
    #             numpyActionVec = actionVec.detach().cpu().numpy()
    #             # pdb.set_trace()
    #             time_start = time.time()
    #             # prev_new_move, move_to_boundary, move_to_wall_agents, collide_agents, collide_in_move_agents = self.check_collision(
    #             #     current_positions, proposed_moves)
    #             self.naiveShieldTime += time.time() - time_start

    #             time_start = time.time()
    #             new_move, move_to_boundary, move_to_wall_agents, collide_agents, collide_in_move_agents = self.lacam_check_collision(
    #                 numpyActionVec, current_positions, agent_priorities)
    #             time_end = time.time()
    #             self.shieldTime += time_end - time_start
    #             # pdb.set_trace()
    #             # numDif = (np.abs(new_move - prev_new_move).sum(axis=-1) > 0).sum()
    #             # if (numDif > 0):
    #             #     print("LaCAM changed {} moves".format(numDif))
    #                 # pdb.set_trace()
    #             # print("Made a move")
            
    #         # Extra check to make sure we didn't break anything!
    #         # new_move, move_to_boundary, move_to_wall_agents, collide_agents, collide_in_move_agents = self.check_collision(
    #         #         self.current_positions, new_move)
    #         # assert(move_to_boundary.sum() + len(move_to_wall_agents) +
    #         #        len(collide_agents) + len(collide_in_move_agents) == 0)

    #         # if not (new_move == proposed_moves).all():
    #         #     print('something changes')

    #         if not check_predictCollsion:
    #             if np.count_nonzero(move_to_boundary) != 0 or len(move_to_wall_agents) != 0 or np.count_nonzero(
    #                     collide_agents) != 0 or len(collide_in_move_agents) != 0:
    #                 check_predictCollsion = True

    #         # if len(move_to_wall_agents) != 0 or np.count_nonzero(collide_agents) != 0 or len(collide_in_move_agents) != 0:
    #         #     print('collision happens step {}'.format(currentstep), move_to_wall_agents, collide_agents, collide_in_move_agents)
    #         # if np.count_nonzero(move_to_boundary) != 0:
    #         #     print('move out of boundaries step {}'.format(currentstep), move_to_boundary)

    #         # Compute Next position
    #         current_positions += new_move
    #         # self.path_list.append(current_positions.copy())

    #         # curLocations = set()
    #         # for agent_id, agent_loc  in enumerate(self.current_positions):
    #         #     if tuple(agent_loc) not in curLocations:
    #         #         curLocations.add(tuple(agent_loc))
    #         #     else:
    #         #         print("UH OH, MULTIPLE AGENTS AT SAME LOCATION!")
    #         #         pdb.set_trace()
    #         # print('decision:', new_move)
    #         # print('new position:', self.current_positions)

    #         # Check reach goals
    #         current_distance = np.sum(np.abs(current_positions - self.goal_positions), axis=1)
    #         # self.reach_goal[current_distance == 0] = 1
    #         # allReachGoalOrig = (np.count_nonzero(self.reach_goal) == self.config.num_agents)
    #         allReachGoal = np.count_nonzero(current_distance) == 0
    #         # if allReachGoalOrig and not allReachGoal:
    #         #     pdb.set_trace()

    #         # pdb.set_trace()
    #         # Update end_step
    #         # self.end_step[(current_distance == 0) & (self.end_step == 0)] = currentstep
    #         ## I think above has a bug when an agent reaches its goal (setting end_step) but then moves out
    #         ## The self.end_step == 0 will be False so it won't be updated
    #         end_step[current_distance != 0] = 0 # Reset end_step for agents that moved out of goal
    #         end_step[(current_distance == 0) & (end_step == 0)] = currentstep

    #     return allReachGoal, check_moveCollision, check_predictCollsion, new_move, end_step

    # def save_success_cases(self, mode):
    #     '''
    #     This function saves the cases into yaml files
    #     Args:
    #         mode: - success: save a tag in the file indicating successful cases.

    #     Returns:

    #     '''
    #     inputfile_name = os.path.join(self.config.result_AnimeDemo_dir_input,
    #                                   'input_map{:02d}x{:02d}_IDMap{:05d}_IDCase{:05d}.yaml'.format(self.size_map[0],
    #                                                                                                 self.size_map[1],
    #                                                                                                 self.ID_MAP,
    #                                                                                                 self.ID_case))
    #     print(inputfile_name)

    #     outputfile_name = os.path.join(self.config.result_AnimeDemo_dir_predict_success,
    #                                    'predict_map{:02d}x{:02d}_IDMap{:05d}_IDCase{:05d}.yaml'.format(self.size_map[0],
    #                                                                                                    self.size_map[1],
    #                                                                                                    self.ID_MAP,
    #                                                                                                    self.ID_case))
    #     if mode == 'success':
    #         checkSuccess = 1
    #     else:
    #         checkSuccess = 0

    #     targetfile_name = os.path.join(self.config.result_AnimeDemo_dir_target,
    #                                    'expert_map{:02d}x{:02d}_IDMap{:05d}_IDCase{:05d}.yaml'.format(self.size_map[0],
    #                                                                                                   self.size_map[1],
    #                                                                                                   self.ID_MAP,
    #                                                                                                   self.ID_case))

    #     gsofile_name = os.path.join(self.config.result_AnimeDemo_dir_GSO,
    #                                 'predict_map{:02d}x{:02d}_IDMap{:05d}_IDCase{:05d}.mat'.format(self.size_map[0],
    #                                                                                                self.size_map[1],
    #                                                                                                self.ID_MAP,
    #                                                                                                self.ID_case))

    #     save_statistics_GSO = {'gso': self.store_GSO, 'attentionGSO': self.store_attentionGSO,
    #                            'commRadius': self.store_communication_radius,
    #                            'FOV_Path': self.store_localPath}

    #     sio.savemat(gsofile_name, save_statistics_GSO)

    #     # print('############## successCases in training set ID{} ###############'.format(self.ID_dataset))
    #     f = open(inputfile_name, 'w')
    #     f.write("map:\n")
    #     f.write("    dimensions: {}\n".format([self.size_map[0], self.size_map[1]]))
    #     f.write("    ID_Map: {}\n".format(self.ID_MAP))
    #     f.write("    ID_Case: {}\n".format(self.ID_case))
    #     f.write("    obstacles:\n")

    #     for ID_obs in range(self.numObstacle):
    #         list_obs = list(self.obstacle_positions[ID_obs])
    #         f.write("    - {}\n".format(list_obs))
    #     f.write("agents:\n")
    #     for id_agent in range(self.config.num_agents):
    #         f.write("  - name: agent{}\n    start: {}\n    goal: {}\n".format(id_agent,
    #                                                                           list(map(int,
    #                                                                                    self.start_positions[id_agent])),
    #                                                                           list(map(int,
    #                                                                                    self.goal_positions[id_agent]))))
    #     f.close()

    #     f_sol = open(outputfile_name, 'w')
    #     f_sol.write("statistics:\n")
    #     f_sol.write("    cost: {}\n".format(int(self.flowtimePredict)))
    #     f_sol.write("    makespan: {}\n".format(int(self.makespanPredict)))
    #     f_sol.write("    succeed: {}\n".format(int(checkSuccess)))
    #     f_sol.write("    lacamShieldTime: {}\n".format(self.shieldTime))
    #     f_sol.write("    naiveShieldTime: {}\n".format(self.naiveShieldTime))
    #     f_sol.write("    totalTime: {}\n".format(self.totalTime))
    #     f_sol.write("schedule:\n")

    #     for id_agent in range(self.config.num_agents):
    #         # print(self.status_MultiAgent[name_agent]["path_predict"])
    #         self.path_matrix = np.array(self.path_list)
    #         len_path = len(self.path_list)

    #         f_sol.write("    agent{}:\n".format(id_agent))
    #         for step in range(len_path):
    #             f_sol.write(
    #                 "       - x: {}\n         y: {}\n         t: {}\n".format(int(self.path_matrix[step][id_agent][0]),
    #                                                                           int(self.path_matrix[step][id_agent][1]),
    #                                                                           step))
    #     f_sol.close()

    #     f_target = open(targetfile_name, 'w')
    #     f_target.write("statistics:\n")
    #     f_target.write("    cost: {}\n".format(int(self.flowtimeTarget)))
    #     f_target.write("    makespan: {}\n".format(int(self.makespanTarget)))
    #     f_target.write("schedule:\n")

    #     for id_agent in range(self.config.num_agents):
    #         len_path = len(self.expert_path_list)
    #         f_target.write("    agent{}:\n".format(id_agent))
    #         for step in range(len_path):
    #             f_target.write("       - x: {}\n         y: {}\n         t: {}\n".format(
    #                 int(self.expert_path_matrix[step][id_agent][0]),
    #                 int(self.expert_path_matrix[step][id_agent][1]),
    #                 step))
    #     f_target.close()

    # def save_failure_cases(self):
    #     '''
    #     This function saves cases (only failed cases) for online expert
    #     Returns:

    #     '''
    #     inputfile_name = os.path.join(self.failureCases_input,
    #                                   'input_map{:02d}x{:02d}_IDMap{:05d}_IDCase{:05d}.yaml'.format(self.size_map[0],
    #                                                                                                 self.size_map[1],
    #                                                                                                 self.ID_MAP,
    #                                                                                                 self.ID_case))
    #     print('############## failureCases in training set ID{} ###############'.format(self.ID_dataset))
    #     f = open(inputfile_name, 'w')
    #     f.write("map:\n")
    #     f.write("    dimensions: {}\n".format([self.size_map[0], self.size_map[1]]))
    #     f.write("    ID_Map: {}\n".format(int(self.ID_MAP)))
    #     f.write("    ID_Case: {}\n".format(int(self.ID_case)))
    #     f.write("    obstacles:\n")
    #     for ID_obs in range(self.numObstacle):
    #         list_obs = list(map(int, self.obstacle_positions[ID_obs]))
    #         f.write("    - {}\n".format(list_obs))

    #     f.write("agents:\n")
    #     for id_agent in range(self.config.num_agents):
    #         f.write("  - name: agent{}\n    start: {}\n    goal: {}\n".format(id_agent,
    #                                                                           list(map(int, self.current_positions[
    #                                                                               id_agent])),
    #                                                                           list(map(int,
    #                                                                                    self.goal_positions[id_agent]))))

    #     f.close()

    # def createfolder_failure_cases(self):
    #     '''
    #     Creat folder for failed cases
    #     Returns:

    #     '''
    #     if os.path.exists(self.failureCases_input) and os.path.isdir(self.failureCases_input):
    #         shutil.rmtree(self.failureCases_input)
    #     if os.path.exists(self.dir_sol) and os.path.isdir(self.dir_sol):
    #         shutil.rmtree(self.dir_sol)
    #     try:
    #         # Create target Directory
    #         os.makedirs(self.failureCases_input)
    #     except FileExistsError:
    #         # print("Directory ", dirName, " already exists")
    #         pass

    # def count_GSO_communcationRadius(self, step):
    #     _ = self.getGSO(step)
    #     _ = self.getCurrentState()
    #     return self.store_GSO, self.store_communication_radius

    # def count_numAgents_ReachGoal(self):
    #     return np.count_nonzero(self.reach_goal)

    # def checkOptimality(self, collisionFreeSol):
    #     '''
    #     Check if the solution is optimal
    #     Args:
    #         collisionFreeSol: - True : solution is free of collision

    #     Returns:

    #     '''
    #     if self.makespanPredict <= self.makespanTarget and self.flowtimePredict <= self.flowtimeTarget and collisionFreeSol:
    #         findOptimalSolution = True
    #     else:
    #         findOptimalSolution = False

    #     return findOptimalSolution, [self.makespanPredict, self.makespanTarget], [self.flowtimePredict,
    #                                                                               self.flowtimeTarget]

    # def get_pos(self, map_tensor):
    #     map_np = map_tensor.numpy()
    #     pos = np.transpose(np.nonzero(map_np))
    #     return pos

    # def findpos(self, channel):
    #     pos_object = channel.nonzero()
    #     num_object = pos_object.shape[0]
    #     pos = torch.zeros(num_object, 2)
    #     # pos_list = []

    #     for i in range(num_object):
    #         pos[i][0] = pos_object[i][0]
    #         pos[i][1] = pos_object[i][1]
    #     #     pos_list.append([pos_object[i][0], pos_object[i][1]])
    #     # pos = torch.FloatTensor(pos_list)
    #     return pos

    # def initCommunicationRadius(self):
    #     self.communicationRadius = self.config.commR

    # def computeAdjacencyMatrix(self, step, agentPos, CommunicationRadius, graphConnected=False):
    #     len_TimeSteps = agentPos.shape[0]  # length of timesteps
    #     nNodes = agentPos.shape[1]  # Number of nodes
    #     # Create the space to hold the adjacency matrices
    #     W = np.zeros([len_TimeSteps, nNodes, nNodes])
    #     W_norm = np.zeros([len_TimeSteps, nNodes, nNodes])
    #     # Initial matrix
    #     distances = squareform(pdist(agentPos[0], 'euclidean'))  # nNodes x nNodes

    #     # I will increase the communication radius by 10% each time,
    #     # but I have to do it consistently within the while loop,
    #     # so in order to not affect the first value set of communication radius, I will account for that initial 10% outside

    #     if step == 0:
    #         self.communicationRadius = self.communicationRadius / 1.1
    #         while graphConnected is False:
    #             self.communicationRadius = self.communicationRadius * 1.1
    #             W[0] = (distances < self.communicationRadius).astype(agentPos.dtype)
    #             W[0] = W[0] - np.diag(np.diag(W[0]))
    #             graphConnected = graph.isConnected(W[0])

    #         # And once we have found a connected initial position, we normalize it
    #         if self.config.symmetric_norm:
    #             deg = np.sum(W[0], axis=1)  # nNodes (degree vector)
    #             zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)[0]
    #             deg[zeroDeg] = 1.
    #             invSqrtDeg = np.sqrt(1. / deg)
    #             invSqrtDeg[zeroDeg] = 0.
    #             Deg = np.diag(invSqrtDeg)
    #             W[0] = Deg @ W[0] @ Deg

    #         maxEigenValue = self.get_maxEigenValue(W[0])
    #         W_norm[0] = W[0] / maxEigenValue
    #     # And once we have found a communication radius that makes the initial graph connected,
    #     # just follow through with the rest of the times, with that communication radius
    #     else:
    #         distances = squareform(pdist(agentPos[0], 'euclidean'))  # nNodes x nNodes
    #         W[0] = (distances < self.communicationRadius).astype(agentPos.dtype)
    #         W[0] = W[0] - np.diag(np.diag(W[0]))
    #         graphConnected = graph.isConnected(W[0])
    #         if np.any(W):
    #             # if W is all non-zero matrix, do normalization
    #             if self.config.symmetric_norm:
    #                 deg = np.sum(W[0], axis=1)  # nNodes (degree vector)
    #                 zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)[0]
    #                 deg[zeroDeg] = 1.
    #                 invSqrtDeg = np.sqrt(1. / deg)
    #                 invSqrtDeg[zeroDeg] = 0.
    #                 Deg = np.diag(invSqrtDeg)
    #                 W[0] = Deg @ W[0] @ Deg
    #             maxEigenValue = self.get_maxEigenValue(W[0])
    #             W_norm[0] = W[0] / maxEigenValue
    #         else:
    #             # if W is all zero matrix, don't do any normalization
    #             W_norm[0] = W

    #     return W_norm, self.communicationRadius, graphConnected

    # def get_maxEigenValue(self, matrix):

    #     isSymmetric = np.allclose(matrix, np.transpose(matrix, axes=[1, 0]))
    #     if isSymmetric:
    #         W = np.linalg.eigvalsh(matrix)
    #     else:
    #         W = np.linalg.eigvals(matrix)

    #     maxEigenvalue = np.max(np.real(W), axis=0)
    #     return maxEigenvalue
    #     # return np.max(np.abs(np.linalg.eig(matrix)[0]))

    # def computeAdjacencyMatrix_fixedCommRadius(self, step, agentPos, CommunicationRadius, graphConnected=False):
    #     len_TimeSteps = agentPos.shape[0]  # length of timesteps
    #     nNodes = agentPos.shape[1]  # Number of nodes
    #     # Create the space to hold the adjacency matrices
    #     W = np.zeros([len_TimeSteps, nNodes, nNodes])
    #     W_norm = np.zeros([len_TimeSteps, nNodes, nNodes])
    #     # Initial matrix
    #     distances = squareform(pdist(agentPos[0], 'euclidean'))  # nNodes x nNodes

    #     W[0] = (distances < self.communicationRadius).astype(agentPos.dtype)
    #     W[0] = W[0] - np.diag(np.diag(W[0]))
    #     graphConnected = graph.isConnected(W[0])
    #     if np.any(W):
    #         # if W is all non-zero matrix, do normalization
    #         if self.config.symmetric_norm:
    #             deg = np.sum(W[0], axis=1)  # nNodes (degree vector)
    #             zeroDeg = np.nonzero(np.abs(deg) < self.zeroTolerance)[0]
    #             deg[zeroDeg] = 1.
    #             invSqrtDeg = np.sqrt(1. / deg)
    #             invSqrtDeg[zeroDeg] = 0.
    #             Deg = np.diag(invSqrtDeg)
    #             W[0] = Deg @ W[0] @ Deg
    #         maxEigenValue = self.get_maxEigenValue(W[0])
    #         W_norm[0] = W[0] / maxEigenValue
    #     else:
    #         # if W is all zero matrix, don't do any normalization
    #         print('No robot are connected at this moment, all zero matrix.')

    #         W_norm[0] = W

    #     return W_norm, self.communicationRadius, graphConnected

    # def getOptimalityMetrics(self):
    #     return [self.makespanPredict, self.makespanTarget], [self.flowtimePredict, self.flowtimeTarget]

    # def getMaxstep(self):
    #     return self.maxstep

    # def getMapsize(self):
    #     return self.size_map

    # def normalize(self, x):

    #     x_normed = x / torch.sum(x, dim=-1, keepdim=True)

    #     return x_normed

    # def convectToActionKey_softmax(self, actionVec):
    #     actionVec_current = self.fun_Softmax(actionVec)
    #     if not self.config.batch_numAgent:
    #         actionKey_predict = torch.max(actionVec_current, 1)[1]
    #     else:
    #         actionKey_predict = torch.max(actionVec_current, 0)[1]
    #     return actionKey_predict

    # def convectToActionKey_sum_multinorm(self, actionVec):
    #     actionVec_current = self.normalize(actionVec)
    #     # print(actionVec_current)
    #     actionKey_predict = torch.multinomial(actionVec_current, 1)[0]

    #     return actionKey_predict

    # def convectToActionKey_exp_multinorm(self, actionVec):
    #     actionVec_current = torch.exp(actionVec)

    #     actionKey_predict = torch.multinomial(actionVec_current, 1)[0]

    #     return actionKey_predict