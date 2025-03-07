
import numpy as np
import scipy
import time

from main import Task

# example of testing a specific task
for id in [5, 10, 25, 40, 50, 80]: 
    T = Task(id)                # initialize task
    T.plan_path()               # path planning
    T.visualize_path()          # path visualization
    print (T.check_path())      # the validity check of the generated path
total_elapsed = 0
for _ in range(10):
    stime = time.time_ns()
    for id in range(100):
        T = Task(id)
        T.plan_path()
        if not T.check_path():
            T.visualize_path()
    etime = time.time_ns()
    total_elapsed += etime-stime
    print(f"Elapsed time in ns: {etime-stime}")
with open("../Project_1_results.txt", 'a') as f:
    f.write(f"Average time in ns for dfs: {total_elapsed / 10}\n")

