import random
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Optional

# --- Game environment ---
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

def check_winner(state):
    current, pursued, pursuer = state
    if current == pursued:
        return 3
    if current == pursuer:
        return -3
    
# --- Q-learning agent ---
Q = defaultdict(float)
alpha = 0.5     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.1   # exploration rate

def stringify(state):
    upper = []
    for s in state:
        sstring = str(s)
        upper.append(sstring)
    return " ".join(upper)

def unstringify(upper):
    state = []
    u = upper.split(" ")
    for s in u:
        pos = s.split(", ")
        state.append(pos)
    return state


def choose_action(world, state, player):
    if random.random() < epsilon:
        return random.choice(get_legal_actions(world, player))
    state_string = stringify(state)
    qs = [Q[(state_string, str(a), str(player))] for a in get_legal_actions(world, player)]
    max_q = max(qs)
    best_actions = [a for a in get_legal_actions(world, player) if Q[(state_string, str(a), str(player))] == max_q]
    return random.choice(best_actions)

# --- Play one game and update Q ---
def play_game(state, rounds=200):
    state = list(state)
    world = state[0]
    state = state[1:]
    for i in range(len(state)):
        state[i] = tuple(state[i])
    player = (0, state[0])
    history = []
    round_num = 0

    while True:
        action = choose_action(world, state, player)
        next_state = apply_action(state, action, player)
        winner = check_winner(next_state)

        history.append((stringify(state), str(action), str(player), stringify(next_state)))

        if winner or round_num < rounds:
            # Assign rewards
            for s, a, p, s_next in reversed(history):
                s_next_real = unstringify(s_next)
                if round_num < rounds:
                    reward = 0
                else:
                    reward = winner 
                max_q = max([Q[(s_next, str(a2), p)] for a2 in get_legal_actions(world, player)])
                Q[(s, a, p)] += alpha * (reward + gamma * max_q - Q[(s, a, p)])
            break

        state = next_state
        player_num = player[0]%3
        player = (player_num, state[player_num])
        round_num += 1

# --- Training loop ---
def train(state):
    for episode in range(1000):
        play_game(state)


class PlannerAgent:
	
    def __init__(self):
        self.first_run = True
        pass
	
    def plan_action(self, world: np.ndarray, current: np.ndarray, pursued: np.ndarray, pursuer: np.ndarray) -> Optional[np.ndarray]:
        """
        Computes a action to take from the current position caputure the pursued while evading from the pursuer

        Parameters:
        - world (np.ndarray): A 2D numpy array representing the grid environment.
        - 0 represents a walkable cell.
        - 1 represents an obstacle.
        - current (np.ndarray): The (row, column) coordinates of the current position.
        - pursued (np.ndarray): The (row, column) coordinates of the agent to be pursued.
        - pursuer (np.ndarray): The (row, column) coordinates of the agent to evade from.

        Returns:
        - np.ndarray: one of the 9 actions from 
                            [0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                            [-1, -1], [-1, 1], [1, -1], [1, 1]
        """
        if self.first_run:
            train((world, current, pursued, pursuer))
        self.first_run = False

        return choose_action(world, [current, pursued, pursuer], (0, current)) 

