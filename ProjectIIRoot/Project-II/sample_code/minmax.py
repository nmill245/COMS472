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

def minimax(state, player, alpha=-float('inf'), beta=float('inf')):
    winner = check_winner(state)
    if winner == 'X':
        return 1, None
    elif winner == 'O':
        return -1, None
    elif winner == 'draw':
        return 0, None

    if player == 'X':
        max_eval = -float('inf')
        best_move = None
        for action in get_legal_actions(state):
            new_state = apply_action(state, action, player)
            eval, _ = minimax(new_state, 'O', alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = action
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move

    else:  # player == 'O'
        min_eval = float('inf')
        best_move = None
        for action in get_legal_actions(state):
            new_state = apply_action(state, action, player)
            eval, _ = minimax(new_state, 'X', alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = action
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def print_board(state):
    b = [c if c != ' ' else '.' for c in state]
    print(f"{b[0]} {b[1]} {b[2]}\n{b[3]} {b[4]} {b[5]}\n{b[6]} {b[7]} {b[8]}\n")

state = ' ' * 9
player = 'X'

while True:
    print_board(state)
    if player == 'X':
        _, move = minimax(state, player)
        print(f"AI chooses: {move}")
    else:
        move = int(input("Your move (0-8): "))
        while state[move] != ' ':
            print("Illegal move, try again.")
            move = int(input("Your move (0-8): "))

    state = apply_action(state, move, player)
    winner = check_winner(state)
    if winner:
        print_board(state)
        print(f"Game over. Winner: {winner}")
        break
    player = 'O' if player == 'X' else 'X'
