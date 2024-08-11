from collections import deque
import time
import psutil
import uuid
def dfs(graph, start, goal):
    stack = [(start, [start])]  # Stack of tuples (node, path)
    visited = set()

    while stack:
        current_node, path = stack.pop()
        if current_node == goal:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append((neighbor, new_path))
    return None

import networkx as nx

def create_networkx_graph(adjacency_dict):
    G = nx.Graph()
    for node, neighbors in adjacency_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G

mac = uuid.getnode()
start_time = time.time()
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

adjacency_dict = {
    '1': ['2', '7','8'],
    '2': ['3', '6'],
    '3': ['4', '5'],
    '8': ['9', '10', '12'],
    '9': ['11'],
    # Add other connections if needed
}

# Create the NetworkX graph
graph_nx = create_networkx_graph(adjacency_dict)

# Calculate and print the path from 1 to 11 using DFS
path_1_to_11 = dfs(graph_nx, '1', '11')
print("Path from 1 to 11:", path_1_to_11)

memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
end_time = time.time()
mac_address = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
print(f"MAC Address: {mac_address}")
print(f"Memory used: {memory_after - memory_before:.2f} MB")
print(f"Time taken: {end_time - start_time:.4f} seconds")
