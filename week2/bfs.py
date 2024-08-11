from collections import deque
import time
import psutil
import uuid
def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])  # Queue of tuples (node, path)

    while queue:
        current_node, path = queue.popleft()
        if current_node == goal:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append((neighbor, new_path))
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
    'B': ['E', 'X'],
    'E': ['P', 'G'],
    'X': ['K', 'A'],
    'P': ['D', 'T'],
    'K': ['M'],
}


graph_nx = create_networkx_graph(adjacency_dict)
path_B_to_M = bfs(graph_nx, 'B', 'M')
print("Path from B to M:", path_B_to_M)
memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
end_time = time.time()
mac_address = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
print(f"MAC Address: {mac_address}")
print(f"Memory used: {memory_after - memory_before:.2f} MB")
print(f"Time taken: {end_time - start_time:.4f} seconds")
