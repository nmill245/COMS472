import math
from typing import List, Tuple, Optional
import numpy as np
import scipy

def score(p1, p2):
    """A fucntion to give L2 loss"""
    p1x, p1y = p1
    p2x, p2y = p2
    return math.sqrt((p1x-p2x)**2+(p1y-p2y)**2)

def priority_sort(cur, end):
    """A priority sort to bring directions that are more often advantageous to the front, hopefully cutting down loops"""
    new_dir = [(-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonal moves
                    (-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    xdir = np.sign(cur[0] - end[0])
    ydir = np.sign(cur[1] - end[1])
    if xdir != 0 and ydir != 0:
        return [(xdir, ydir),(xdir, 0), (ydir, 0), (-xdir, ydir), (xdir, -ydir), (-xdir, -ydir), (-xdir, 0), (-ydir, 0)]
    if ydir != 0:
        return [(0, ydir), (-1, ydir), (1, ydir), (1, -ydir),(0, -ydir), (-1, -ydir), (1, 0), (-1, 0)]
    if xdir != 0:
        return [(xdir, 0), (xdir, -1), (xdir, 1), (-xdir, 1),(-xdir, 0), (-xdir, -1), (0, 1), (0, -1)]
    return new_dir


def a_star(grid, start, end):
    """An A* implementation"""
    #generate placeholders
    rows, cols = len(grid), len(grid[0])
    directions =  [(-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonal moves
                    (-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    visited = {}
    moves = {}
    reached = False

    #generate start point as initial point
    p1 = start
    visited[start] = (0, [start])


    while not reached:
        move = (0, 0)
        min_overall_cost = math.inf
        fpath = []
        for p1, (fcost, point_path) in visited.items():
            min_move_cost = math.inf
            if p1 in moves:
                fcost, move_path, mmove = moves[p1]
                if fcost >= min_overall_cost or mmove in visited:
                    continue
                fpath = []
                for p in move_path:
                    fpath.append(p)
                move = mmove
                min_overall_cost = fcost
                continue
            #if the cost is over the min already, don't need to explore branch
            if fcost >= min_overall_cost:
                continue
            p1x, p1y = p1
            directions = priority_sort(p1, end)
            for dirc in directions:
                dx, dy = dirc
                p2x = p1x + dx
                p2y = p1y + dy
                #check if move is valid and unmade
                if p2x >= rows or p2y >= cols or p2x < 0 or p2y < 0 or grid[p2x][p2y] == 1 or (p2x, p2y) in visited.keys():
                    continue
                cost = fcost + score((p2x, p2y), end)
                #updating global mins
                if cost < min_overall_cost:
                    min_overall_cost = cost
                    move = (p2x, p2y)
                    #get total path to inital point and store it
                    fpath = []
                    for p in point_path:
                        fpath.append(p)
                #update point specific min
                if cost < min_move_cost:
                    min_move_cost = cost
                    mpath = []
                    for p in point_path:
                        mpath.append(p)
                    moves[p1] = (min_move_cost, mpath, (p2x, p2y))
        #take the move
        fpath.append(move)
        visited[move] = (min_overall_cost, fpath)
        if move == end:
            reached = True
            return fpath
    return None

def plan_path(world: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[np.ndarray]:
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
    # Ensure start and end positions are tuples of integers
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))

    # Convert the numpy array to a list of lists for compatibility with the example DFS function
    world_list: List[List[int]] = world.tolist()

    # Perform DFS pathfinding and return the result as a numpy array
    #path = dfs(world_list, start, end)
    path = a_star(world_list, start, end)
    return np.array(path) if path else None
