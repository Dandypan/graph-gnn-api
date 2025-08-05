import networkx as nx
from torch_geometric.data import Data
import torch
import random


def load_default_graph(noise_1=0.0, noise_2=0.0) -> Data:
    G = nx.Graph()

    # Define cliques
    clique1 = [f"n{i}" for i in range(10)]
    clique2 = [f"n{i}" for i in range(10, 25)]
    clique3 = [f"n{i}" for i in range(25, 37)]
    clique4 = [f"n{i}" for i in range(37, 45)]
    clique5 = [f"n{i}" for i in range(45, 65)]

    cliques = [clique1, clique2, clique3, clique4, clique5]
    for clique in cliques:
        G.add_nodes_from(clique)
        G.add_edges_from([(u, v) for i, u in enumerate(clique) for v in clique[i + 1:]])

    # Add bridges between cliques
    bridges = [(clique1[0], clique2[0]), (clique2[-1], clique3[0]),
               (clique3[-1], clique4[0]), (clique4[-1], clique5[0])]
    G.add_edges_from(bridges)

    # Apply node relabeling for tensor indexing
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Add noise_1 type: random noise edges
    if noise_1 > 0.0:
        total_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
        existing_edges = G.number_of_edges()
        max_noise_edges = total_possible - existing_edges
        noise_edges_to_add = int(noise_1 * G.number_of_edges())

        added = 0
        while added < noise_edges_to_add and added < max_noise_edges:
            u, v = random.sample(range(G.number_of_nodes()), 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1

    # Create initial features and labels
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    x = torch.ones((G.number_of_nodes(), 10), dtype=torch.float)
    y = torch.randint(0, 2, (G.number_of_nodes(),), dtype=torch.long)

    # Add noise_2 type: flip random labels
    if noise_2 > 0.0:
        num_flips = int(noise_2 * len(y))
        flip_indices = random.sample(range(len(y)), num_flips)
        for idx in flip_indices:
            y[idx] = 1 - y[idx]  # For binary labels: 0 <-> 1

    return Data(x=x, edge_index=edge_index, y=y)
