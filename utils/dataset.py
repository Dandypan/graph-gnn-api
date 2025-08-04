import networkx as nx
from torch_geometric.data import Data
import torch


def  load_default_graph() -> Data:
    G = nx.Graph()

    # Clique 1: 10 nodes
    clique1 = [f"n{i}" for i in range(10)]
    G.add_nodes_from(clique1)
    G.add_edges_from([(u, v) for i, u in enumerate(clique1) for v in clique1[i + 1:]])

    # Clique 2: 15 nodes
    clique2 = [f"n{i}" for i in range(10, 25)]
    G.add_nodes_from(clique2)
    G.add_edges_from([(u, v) for i, u in enumerate(clique2) for v in clique2[i + 1:]])

    # Clique 3: 12 nodes
    clique3 = [f"n{i}" for i in range(25, 37)]
    G.add_nodes_from(clique3)
    G.add_edges_from([(u, v) for i, u in enumerate(clique3) for v in clique3[i + 1:]])

    # Clique 4: 8 nodes
    clique4 = [f"n{i}" for i in range(37, 45)]
    G.add_nodes_from(clique4)
    G.add_edges_from([(u, v) for i, u in enumerate(clique4) for v in clique4[i + 1:]])

    # Clique 5: 20 nodes
    clique5 = [f"n{i}" for i in range(45, 65)]
    G.add_nodes_from(clique5)
    G.add_edges_from([(u, v) for i, u in enumerate(clique5) for v in clique5[i + 1:]])

    # Connect cliques with bridges
    bridges = [(clique1[0], clique2[0]), (clique2[-1], clique3[0]), (clique3[-1], clique4[0]), (clique4[-1], clique5[0])]
    G.add_edges_from(bridges)

    # Add some noise edges
    import random
    all_nodes = list(G.nodes())
    for _ in range(50):
        u, v = random.sample(all_nodes, 2)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    x = torch.ones((G.number_of_nodes(), 10), dtype=torch.float)
    y = torch.randint(0, 2, (G.number_of_nodes(),), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data
