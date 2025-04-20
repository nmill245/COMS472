import math
import random
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Optional

def get_all_legal_actions(world, state, current_player_num):
    all = []
    for i in range(len(state)):
        if i == current_player_num:
            continue
        actions = get_legal_actions(world, (-1, state[i]))
        all.extend(actions)
    return all

def get_legal_actions(world, player):
    directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                           [-1, -1], [-1, 1], [1, -1], [1, 1]]) 
    current = player[1]
    rows = len(world)
    cols = len(world[0])
    legal = []
    for dirc in directions:
        nx = current[0] + dirc[0]
        ny = current[1] + dirc[1]
        if 0 <= nx < rows and 0 <= ny < cols and world[nx][ny] == 0:
            legal.append(dirc)
    return legal

def apply_action(state, action, player):
    state[player[0]] = (player[1][0] + action[0], player[1][1] + action[1]) 
    return state

def check_winner(state, player):
    current = state[player[0]]
    pursued = state[(player[0]-1)%3]
    pursuer = state[(player[0]+1)%3]
    cur_win = current == pursued
    cur_lose = current == pursuer
    pur_win = pursued == pursuer
    if cur_win and not cur_lose and not pur_win:
        return 3
    if cur_lose and not cur_win and not pur_win:
        return -3 
    if pur_win and not cur_win and not cur_lose:
        return -1
    if cur_win or cur_lose or pur_win:
        return 1

# --- MCTS Node ---
class Node:
    def __init__(self, world, state, player, parent=None):
        self.world = world
        self.state = state
        self.player = player
        self.parent = parent
        self.children = {}  # action -> Node
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(get_all_legal_actions(self.world, self.state, self.player))

    def best_child(self, c=1.41):
        return max(
            self.children.items(),
            key=lambda item: item[1].value / (item[1].visits + 1e-6) +
                             c * math.sqrt(math.log(self.visits + 1) / (item[1].visits + 1e-6))
        )[1]

    def expand(self):
        for action in get_legal_actions(self.world, self.player):
            str_action = str(action)
            if str_action not in self.children:
                next_state = apply_action(self.state, action, self.player)
                next_player_num = (self.player[0]+1)%3
                next_player = (next_player_num, self.state[next_player_num])
                child = Node(self.world, next_state, next_player, parent=self)
                self.children[str_action] = child
                return child
        return self  # fallback (shouldn’t happen)

    def rollout(self, rounds=50):
        state = self.state
        player = self.player
        i = 0
        while True:
            winner = check_winner(state, player)
            if winner:
                return winner
            actions = get_legal_actions(state, player)
            if i < rounds:
                return 0
            action = random.choice(actions)
            state = apply_action(state, action, player)
            player_num = (player[0]+1)%3
            player = (player_num, self.state[player_num])
            i += i

    def backpropagate(self, result, root_player):
        reward = result
        node = self
        results = [3, -3, -1]
        i = -1
        if not (reward in (0, -1)):
            i = results.index(reward)
        while node:
            node.visits += 1
            node.value += reward
            if not (reward in (0, -1)):
                i = (i+1)%3
                reward = results[i] 
            node = node.parent

# --- MCTS search ---
def mcts_search(world, state, player, simulations=1000):
    root = Node(world, state, player)
    for _ in range(simulations):
        node = root

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion
        if not check_winner(node.state, node.player):
            node = node.expand()

        # Simulation
        result = node.rollout()

        # Backpropagation
        node.backpropagate(result, root.player)

    # Pick action with highest visit count
    return max(root.children.items(), key=lambda item: item[1].visits)[0]
class PlannerAgent:
	
    def __init__(self):
        pass

    def plan_action(self, world: np.ndarray, current: Tuple[int, int], pursued: Tuple[int, int], pursuer: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Computes a path from the start position to the end position 
        using a certain planning algorithm (DFS is provided as an example).

        Parameters:
        - world (np.ndarray): A 2D numpy array representing the grid environment.
        - 0 represents a walkable cell.
        - 1 represents an obstacle.
        - start (Tuple[int, int]): The (row, column) coordinates of the starting position.
        - end (Tuple[int, int]): The (row, column) coordinates of the goal position.

        Returns:
        - np.ndarray: A 2D numpy array where each row is a (row, column) coordinate of the path.
        The path starts at 'start' and ends at 'end'. If no path is found, returns None.
        """

        str_action = mcts_search(world, [tuple(current), tuple(pursued), tuple(pursuer)], (0, current), simulations=100)
        action = str_action.replace("[", "").replace("]", "").split(" ")
        final_action = []
        for a in action:
            final_action.append(int(a))
        action = np.array(final_action)
        return action


