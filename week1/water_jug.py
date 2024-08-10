from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import uuid
queue = deque()
visited = set()
queue.append((0, 0,[(0,0)]))

def problem(jug1, jug2, q):
    while queue:
        j1, j2 ,path= queue.popleft()
        if (j1, j2) in visited:
            continue
        if j1 == q:
            print("Solution Found")
            return path
        visited.add((j1, j2))
        queue.append((jug1, j2,path+[(jug1, j2)]))
        queue.append((j1, jug2,path+[(j1, jug2)]))
        queue.append((0, j2,path+[(0,j2)]))
        queue.append((j1, 0,path+[(j1,0)]))
       
        tf = min(j1, jug2 - j2)
        queue.append((j1 - tf, j2 + tf,path+[(j1 - tf, j2 + tf)]))
        tf = min(j2, jug1 - j1)
        queue.append((j1 + tf, j2 - tf,path+[(j1 + tf, j2 - tf)]))
    return "NO SOLUTION"


mac = uuid.getnode()
start_time = time.time()
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
jug1 = int(input("Enter the quantity of Jug 1: "))
jug2 = int(input("Enter the quantity of Jug 2: "))   
qty = int(input("Enter the quantity required in Jug 1: "))   

print(problem(jug1, jug2, qty))
memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
end_time = time.time()
mac_address = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
print(f"MAC Address: {mac_address}")
print(f"Memory used: {memory_after - memory_before:.2f} MB")
print(f"Time taken: {end_time - start_time:.4f} seconds")
