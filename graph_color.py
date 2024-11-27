import networkx as nx

def create_graph(l):
    G = nx.Graph()
    for node, neighbors in l.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G

colors = ['red','green','blue']
l = {
    'wa': ['nt', 'sa'],
    'nt': ['q', 'sa'],
    'sa': ['q', 'nsw', 'v'],
    'q': ['nsw'],
    'nsw': ['v'],
    't': []
}

G = create_graph(l)

def sat(G):
    for i in G.nodes:
        if G.nodes[i]['color'] != '':
            for j in G.neighbors(i):
                if G.nodes[j]['color'] == G.nodes[i]['color'] and i != j:
                    return False
    return True

def emp(G):
    for i in G.nodes:
        if G.nodes[i]['color'] == '':
            return True
    return False

def empty(c):
    for i in c:
        if c[i] == "":
            return i
    return None

def bfs(G):
    visited = set()
    initial_colors = {node: '' for node in G.nodes}
    stack = [initial_colors]  # Initial coloring
    while stack:
        current_color = stack.pop()
        # Set the colors in G according to the current coloring
        nx.set_node_attributes(G, current_color, name="color")
        
        if tuple(current_color.items()) in visited:
            continue
        visited.add(tuple(current_color.items()))
        
        # Check if current coloring is valid and all nodes are colored
        if sat(G) and not emp(G):
            return current_color
        
        # Find the first uncolored node
        node_to_color = empty(current_color)
        if node_to_color is None:
            continue

        # Try coloring the node with all possible colors
        for color in colors:
            new_color = current_color.copy()
            new_color[node_to_color] = color
            if sat_color(G, new_color, node_to_color, color):
                stack.append(new_color)

    return None

def sat_color(G, color_dict, node, color):
    # Temporarily assign color and check validity
    G.nodes[node]['color'] = color
    valid = sat(G)
    G.nodes[node]['color'] = ''  # Unassign color
    return valid

# Initialize node attributes to empty color
initial_colors = {node: '' for node in G.nodes}
nx.set_node_attributes(G, initial_colors, name="color")

solution = bfs(G)
if solution:
    print("Valid coloring found:")
    print(solution)
else:
    print("No valid coloring found.")
