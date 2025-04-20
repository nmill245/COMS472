import random
from collections import defaultdict

# --- Game environment ---
def get_legal_actions(state):
    return [i for i in range(9) if state[i] == ' ']

def apply_action(state, action, player):
    assert state[action] == ' ', f"Illegal move: cell {action} is already occupied"
    return state[:action] + player + state[action+1:]

def check_winner(state):
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for i,j,k in lines:
        if state[i] == state[j] == state[k] and state[i] != ' ':
            return state[i]
    return 'draw' if ' ' not in state else None

# --- Q-learning agent ---
Q = defaultdict(float)
alpha = 0.5     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.1   # exploration rate

def choose_action(state, player):
    if random.random() < epsilon:
        return random.choice(get_legal_actions(state))
    qs = [Q[(state, a, player)] for a in get_legal_actions(state)]
    max_q = max(qs)
    best_actions = [a for a in get_legal_actions(state) if Q[(state, a, player)] == max_q]
    return random.choice(best_actions)

# --- Play one game and update Q ---
def play_game():
    state = ' ' * 9
    player = 'X'
    history = []

    while True:
        action = choose_action(state, player)
        next_state = apply_action(state, action, player)
        winner = check_winner(next_state)

        history.append((state, action, player, next_state))

        if winner:
            # Assign rewards
            for s, a, p, s_next in reversed(history):
                if winner == 'draw':
                    reward = 0
                else:
                    reward = 1 if p == winner else -1
                max_q = max([Q[(s_next, a2, p)] for a2 in get_legal_actions(s_next)] + [0])
                Q[(s, a, p)] += alpha * (reward + gamma * max_q - Q[(s, a, p)])
            break

        state = next_state
        player = 'O' if player == 'X' else 'X'

# --- Training loop ---
for episode in range(10000):
    play_game()

print (Q)

# --- Evaluate learned policy ---
def print_board(state):
    board = [c if c != ' ' else '.' for c in state]
    print(f"{board[0]} {board[1]} {board[2]}\n{board[3]} {board[4]} {board[5]}\n{board[6]} {board[7]} {board[8]}\n")
"""
state = ' ' * 9
player = 'X'
while True:
    print_board(state)
    if player == 'X':
        action = choose_action(state, player)
    else:
        action = int(input("Your move (0-8): "))
    state = apply_action(state, action, player)
    winner = check_winner(state)
    if winner:
        print_board(state)
        print(f"Winner: {winner}")
        break
    player = 'O' if player == 'X' else 'X'
"""
