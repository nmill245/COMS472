import numpy as np
from typing import List, Tuple, Optional
"""
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


def minimax(state, player, alpha=-float('inf'), beta=float('inf')):
    winner = check_winner(state)
    if winner:
        return winner, None

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


def dfs(grid, start, end):
    """A DFS example"""
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    visited = set()
    parent = {start: None}

    # Consider all 8 possible moves (up, down, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal moves

    while stack:
        x, y = stack.pop()
        if (x, y) == end:
            # Reconstruct the path
            path = []
            while (x, y) is not None:
                path.append((x, y))
                if parent[(x, y)] is None:
                    break  # Stop at the start node
                x, y = parent[(x, y)]
            return path[::-1]  # Return reversed path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and (nx, ny) not in visited:
                stack.append((nx, ny))
                parent[(nx, ny)] = (x, y)

    return None
"""

class PlannerAgent:
    
    def __init__(self):
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
        
        directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                   [-1, -1], [-1, 1], [1, -1], [1, 1]])
          
        # Ensure start and end positions are tuples of integers
        start = (int(current[0]), int(current[1]))
        end = (int(pursued[0]), int(pursued[1]))

        # Convert the numpy array to a list of lists for compatibility with the example DFS function
        world_list: List[List[int]] = world.tolist()

        # Perform DFS pathfinding and return the result as a numpy array
        path = dfs(world_list, start, end)


        try:
            return np.array(path)[1]-current
        except:
            return directions[np.random.choice(9)]


