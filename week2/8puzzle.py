from collections import deque
import time
import psutil
import uuid
def bfs(start, goal):
    def get_neighbors(state):
        neighbors = []
        index = state.index(0)
        row, col = divmod(index, 3)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for move in moves:
            new_row, new_col = row + move[0], col + move[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                new_state = state[:]
                new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
                neighbors.append(new_state)
        return neighbors

    queue = deque([(start, [])])
    visited = set()
    visited.add(tuple(start))

    while queue:
        current_state, path = queue.popleft()
        if current_state == goal:
            return path + [current_state]
        for neighbor in get_neighbors(current_state):
            if tuple(neighbor) not in visited:
                visited.add(tuple(neighbor))
                queue.append((neighbor, path + [current_state]))
    return None

def print_board(state):
    for i in range(3):
        for j in range(3):
            if state[i * 3 + j] == 0:
                print(" ", end=" ")
            else:
                print(state[i * 3 + j], end=" ")
        print()  
    print()  

mac = uuid.getnode()
start_time = time.time()
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
initial_state = [2, 8, 3, 1, 6, 4, 7, 0, 5]
goal_state = [1, 2, 3, 8, 0, 4, 7, 6, 5]

# Solve the puzzle
solution_path = bfs(initial_state, goal_state)
if solution_path:
    for step in solution_path:
        print_board(step)
else:
    print("No solution found.")
    
memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
end_time = time.time()
mac_address = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
print(f"MAC Address: {mac_address}")
print(f"Memory used: {memory_after - memory_before:.2f} MB")
print(f"Time taken: {end_time - start_time:.4f} seconds")
