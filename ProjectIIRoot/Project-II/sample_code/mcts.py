import math
import random
from collections import defaultdict

# --- Game logic ---
def get_legal_actions(state):
    return [i for i in range(9) if state[i] == ' ']

def apply_action(state, action, player):
    assert state[action] == ' ', "Tried to overwrite a filled square!"
    return state[:action] + player + state[action+1:]

def check_winner(state):
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for i,j,k in lines:
        if state[i] == state[j] == state[k] and state[i] != ' ':
            return state[i]
    return 'draw' if ' ' not in state else None

# --- MCTS Node ---
class Node:
    def __init__(self, state, player, parent=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = {}  # action -> Node
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(get_legal_actions(self.state))

    def best_child(self, c=1.41):
        return max(
            self.children.items(),
            key=lambda item: item[1].value / (item[1].visits + 1e-6) +
                             c * math.sqrt(math.log(self.visits + 1) / (item[1].visits + 1e-6))
        )[1]

    def expand(self):
        for action in get_legal_actions(self.state):
            if action not in self.children:
                next_state = apply_action(self.state, action, self.player)
                next_player = 'O' if self.player == 'X' else 'X'
                child = Node(next_state, next_player, parent=self)
                self.children[action] = child
                return child
        return self  # fallback (shouldn’t happen)

    def rollout(self):
        state = self.state
        player = self.player
        while True:
            winner = check_winner(state)
            if winner:
                return winner
            actions = get_legal_actions(state)
            if not actions:
                return 'draw'
            action = random.choice(actions)
            state = apply_action(state, action, player)
            player = 'O' if player == 'X' else 'X'

    def backpropagate(self, result, root_player):
        reward = 1 if result == root_player else 0 if result == 'draw' else -1
        node = self
        while node:
            node.visits += 1
            node.value += reward
            reward = -reward  # opponent’s turn
            node = node.parent

# --- MCTS search ---
def mcts_search(state, player, simulations=1000):
    root = Node(state, player)
    for _ in range(simulations):
        node = root

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion
        if not check_winner(node.state):
            node = node.expand()

        # Simulation
        result = node.rollout()

        # Backpropagation
        node.backpropagate(result, root.player)

    # Pick action with highest visit count
    return max(root.children.items(), key=lambda item: item[1].visits)[0]

# --- Game loop: MCTS vs human ---
def print_board(state):
    b = [c if c != ' ' else '.' for c in state]
    print(f"{b[0]} {b[1]} {b[2]}\n{b[3]} {b[4]} {b[5]}\n{b[6]} {b[7]} {b[8]}\n")

state = ' ' * 9
player = 'X'

while True:
    print_board(state)
    if player == 'X':
        action = mcts_search(state, player, simulations=300)
        print(f"MCTS chooses: {action}")
    else:
        action = int(input("Your move (0–8): "))
        while state[action] != ' ':
            print("Illegal move, try again.")
            action = int(input("Your move (0–8): "))

    state = apply_action(state, action, player)
    winner = check_winner(state)
    if winner:
        print_board(state)
        print("Winner:", winner)
        break
    player = 'O' if player == 'X' else 'X'
