from queue import PriorityQueue
import pdb
from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np

class EnvironmentWrapper(ABC):
    def __init__(self, grid_occupancy) -> None:
        super().__init__()
        self.grid_occupancy = grid_occupancy

    def getValid(self, row, col):
        g_row, g_col = self.grid_occupancy.shape
        if row>=g_row or col>=g_col or row<0 or col<0:
            return False
        return self.grid_occupancy[row, col] == 0


class AbstractNode(ABC):
    def __init__(self, state, parent) -> None:
        super().__init__()
        self.state = state
        self.parent = parent

    @abstractmethod
    def getSuccessors(self) -> list["AbstractNode"]:
        # Need "AbstractNode" via https://github.com/microsoft/pylance-release/issues/898
        pass

    @abstractmethod
    def getF(self) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def getState(self):
        return self.state

    def getParent(self) -> "AbstractNode":
        return self.parent

    def __lt__(self, other) -> bool:
        return self.getF() < other.getF()

class GenericPlanner:
    def __init__(self, environment: EnvironmentWrapper):
        self.environment = environment

    def findPlan(self, startNode: AbstractNode, goalFunc: Callable[[AbstractNode], bool]):
                    # getSuccessors: Callable[[AbstractNode], list[AbstractNode]]):
        open_pq = PriorityQueue()
        open_pq.put(startNode)
        closed_set = set()

        num_expansions = 0
        while not open_pq.empty():
            current_node : AbstractNode = open_pq.get() # Note get pops the element as well
            if num_expansions > 5000000:
                # print("Exceeded Expansion Limit")
                break
            # print(f"Size: {open_pq.qsize()}")

            if current_node.getState() in closed_set:
                continue
            closed_set.add(current_node.getState())
            # if num_expansions % 10000 == 0:
                # print(num_expansions, current_node)

            if goalFunc(current_node):
                # print(f"Found path after {num_expansions} expansions")
                return self.reconstructPath(current_node)

            num_expansions += 1
            # pdb.set_trace()
            # for next_node in getSuccessors(current_node):
            for next_node in current_node.getSuccessors():
                open_pq.put(next_node)
        # if open_pq.empty():
            # print(f"Emptied the queue after {num_expansions} expansions")
        return None

    def reconstructPath(self, node: AbstractNode):
        """Backtracks to reconstruct the path from the start node to the current node"""
        path = []
        while node is not None:
            path.append(node.getState())
            node = node.getParent()
        path.reverse()
        return path



def computeHeuristicMap(env: EnvironmentWrapper, goal_xy):
    """Precompute the heuristic for all states"""
    goal_xy = tuple(goal_xy)
    assert(len(goal_xy) == 2)
    heuristic_map = np.zeros(env.grid_occupancy.shape)

    class XYNode(AbstractNode):
        def __init__(self, state, parent, g_val) -> None:
            super().__init__(state=state, parent=parent)
            self.g_val = g_val

        # Override
        def getF(self) -> float:
            return self.g_val

        def __str__(self) -> str:
            return f"State: {self.state}, g: {self.g_val}"

        # Override
        def getSuccessors(self) -> list[AbstractNode]:
            state = self.getState()
            assert(len(state) == 2)
            heuristic_map[state[0], state[1]] = self.g_val  # Set the heuristic value
            nextNodes = []
            FOUR_CONNECTED = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dState in FOUR_CONNECTED:
                new_state = (state[0] + dState[0], state[1] + dState[1])
                if env.getValid(new_state[0], new_state[1]):
                    next_node = XYNode(new_state, self, self.g_val+1)
                    nextNodes.append(next_node)
            return nextNodes

    gp = GenericPlanner(environment=env)
    goalFunc = lambda node: False
    startNode = XYNode(goal_xy, None, 0)
    gp.findPlan(startNode, goalFunc)
    return heuristic_map
