import random
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Optional

class PlannerAgent:
    def __init__(self):
        self.Q = defaultdict(float)
        self.alpha = 0.5     # learning rate
        self.gamma = 0.9     # discount factor
        self.epsilon = 0.1   # exploration rate
        self.prob = [0.3, 0.3, 0.4] #probability distrobution
        self.mod_counter = [1, 1, 1] #counter for each of the three mod events
        self.runs = 3 #total number of trials that have had affects on mod_counter
        self.count = 0 #total number of rounds
        self.given = np.array([0, 0]) #the last given action
        self.prev_state = np.array([0, 0])
        self.first_run = True

    def mod_action(self, a):
        """
        Implementation of a random chance for the action to be changed
        """
        mod = np.random.choice(3, p=self.prob)
        match (mod):
            case 0:
                return a
            case 1:
                return np.array([-a[1], a[0]])
            case _:
                return np.array([a[1], -a[0]])

    def update_prob(self, actual):
        """
        A function to update the probability model in the planner agent based off the actual modulation of the actions
        """
        index = -1
        update = False
        #get what modulation actually happened
        if (actual == self.given).all():
            index = 0
            update = True
        if (actual == np.array([-self.given[1], self.given[0]])).all():
            index = 1
            update = True
        if (actual == np.array([self.given[1], -self.given[0]])).all():
            index = 2
            update = True
        if index == -1:
            self.runs -= 1
            """
            if (actual == np.array([0, 0])).all():
                if self.mod_counter[0] > 0:
                    self.mod_counter[0] -= 1
                    self.runs -= 1
                    update = True
            """
        if update:
            self.mod_counter[index] += 1
            for i, _ in enumerate(self.prob):
                self.prob[i] = self.mod_counter[i] / self.runs

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
        a = self.mod_action(action)
        state[player[0]] = (player[1][0] + a[0], player[1][1] + a[1])
        return state

    def check_winner(self, state, player, world):
        """
        Checks if there has been a winner
        Returns the score regarding the player for the winner
        """
        current = state[player[0]]
        pursued = state[(player[0]-1)%3]
        pursuer = state[(player[0]+1)%3]
        rows = len(world)
        cols = len(world[0])
        if current == pursued:
            return 3
        if current == pursuer:
            return -3
        if pursued == pursuer:
            return 1
        if current[0] >= rows or current[1] >= cols or world[current[0]][current[1]] == 1:
            return -5
        if pursued[0] >= rows or pursued[1] >= cols or world[pursued[0]][pursued[1]] == 1:
            return 1
        if pursuer[0] >= rows or pursuer[1] >= cols or world[pursuer[0]][pursuer[1]] == 1:
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
        k = len(self.get_legal_actions(world, player))
        if random.random() < self.epsilon:
            if k < 1:
                return np.array([0, 0])
            return random.choice(self.get_legal_actions(world, player))
        #must use string of state in order to store as dictionary key
        state_string = self.stringify(state)
        qs = [self.Q[(state_string, str(a), str(player))] for a in self.get_legal_actions(world, player) + [0]]
        max_q = max(qs)
        best_actions = [a for a in self.get_legal_actions(world, player) if self.Q[(state_string, str(a), str(player))] == max_q]
        k = len(best_actions)
        if k == 0:
            return np.array([0, 0])
        return random.choice(best_actions)

    # --- Play one game up to 100 actions and update Q ---
    def play_game(self, state, rounds=10):
        """
        A function to play a game up to rounds number of rounds or until
        someone wins
        """
        #assure state is a list
        state = list(state)

        #print("Game Start")

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
            winner = self.check_winner(next_state, player, world)

            history.append((self.stringify(state), str(action), player, self.stringify(next_state)))

            if winner or round_num > rounds:
                # Assign rewards
                for s, a, p, s_next in reversed(history):
                    if round_num > rounds:
                        reward = 1
                    else:
                        reward = winner
                    max_q = max([self.Q[(s_next, str(a2), p)] for a2 in self.get_legal_actions(world, p)] + [0])
                    self.Q[(s, a, p)] += self.alpha * (reward + self.gamma * max_q - self.Q[(s, a, p)])
                break

            state = next_state
            player_num = (player[0]+1)%3
            player = (player_num, state[player_num])
            round_num += 1
        #print("Game Done")

    # --- Training loop ---
    def train(self, state):
        """
        Run a training loop of multiple games to cement the Q table
        """
        for _ in range(100):
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
        self.count += 1
        if not self.first_run:
            self.runs += 1
            self.update_prob(current - self.prev_state)
        self.first_run = False
        if self.count % 10 == 0:
            self.train((world, current, pursued, pursuer))

        action = self.choose_action(world, [current, pursued, pursuer], (0, current))
        self.given = action
        self.prev_state = current
        return action
