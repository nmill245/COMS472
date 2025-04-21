import random
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Optional

class PlannerAgent:
    def __init__(self):
        self.first_run = True
        self.Q = defaultdict(float)
        self.alpha = 0.5     # learning rate
        self.gamma = 0.9     # discount factor
        self.epsilon = 0.1   # exploration rate
	
    def get_legal_actions(self, world, player):
        """
        A function to get all legal actions regarding a player
        Returns an array of (x, y) that are valid moves
        """
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

    def apply_action(self, state, action, player):
        """
        A function to move the player with the given action
        Returns the updated state
        """
        state[player[0]] = (player[1][0] + action[0], player[1][1] + action[1]) 
        return state

    def check_winner(self, state, player):
        """
        Checks if there has been a winner
        Returns the score regarding the player for the winner
        """
        current = state[player[0]]
        pursued = state[(player[0]-1)%3]
        pursuer = state[(player[0]+1)%3]
        if current == pursued:
            return 3
        if current == pursuer:
            return -3
        if pursued == pursuer:
            return 1
        
    # --- Q-learning agent ---

    def stringify(self, state):
        """
        A function that converts the state of players positions into a string
        Returns the converted string
        """
        upper = []
        for s in state:
            sstring = str(s)
            upper.append(sstring)
        return " ".join(upper)

    def unstringify(self, upper):
        """
        A function that converts the string of the state into an array of positions
        Returns the array
        """
        state = []
        u = upper.split(" ")
        for s in u:
            pos = s.split(", ")
            state.append(pos)
        return state


    def choose_action(self, world, state, player):
        """
        A function to choose an action given a specific state
        """
        #keep a chance to take a random action
        if random.random() < self.epsilon:
            return random.choice(self.get_legal_actions(world, player))
        #must use string of state in order to store as dictionary key
        state_string = self.stringify(state)
        qs = [self.Q[(state_string, str(a), str(player))] for a in self.get_legal_actions(world, player)]
        max_q = max(qs)
        best_actions = [a for a in self.get_legal_actions(world, player) if self.Q[(state_string, str(a), str(player))] == max_q]
        return random.choice(best_actions)

    # --- Play one game up to 200 and update Q ---
    def play_game(self, state, rounds=200):
        """
        A function to play a game up to rounds number of rounds or until
        someone wins
        """
        #assure state is a list
        state = list(state)

        #get the map, and all the positions of players
        world = state[0]
        state = state[1:]

        for i in range(len(state)):
            state[i] = tuple(state[i])
        #start with "current"
        player = (0, state[0])
        history = []
        round_num = 0

        while True:

            action = self.choose_action(world, state, player)
            next_state = self.apply_action(state, action, player)
            winner = self.check_winner(next_state, player)

            history.append((self.stringify(state), str(action), str(player), self.stringify(next_state)))

            if winner or round_num < rounds:
                # Assign rewards
                for s, a, p, s_next in reversed(history):
                    if round_num < rounds:
                        reward = 0
                    else:
                        reward = winner
                    max_q = max([self.Q[(s_next, str(a2), p)] for a2 in self.get_legal_actions(world, player)])
                    self.Q[(s, a, p)] += self.alpha * (reward + self.gamma * max_q - self.Q[(s, a, p)])
                break

            state = next_state
            player_num = (player[0]+1)%3
            player = (player_num, state[player_num])
            round_num += 1

    # --- Training loop ---
    def train(self, state):
        for _ in range(1000):
            self.play_game(state)


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
            self.train((world, current, pursued, pursuer))
        self.first_run = False

        action = self.choose_action(world, [current, pursued, pursuer], (0, current))
        return action
