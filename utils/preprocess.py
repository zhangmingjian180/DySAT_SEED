import torch
import torch_geometric

import networkx as nx
import pickle
import numpy as np


def dynamic_graph_to_pyg(dynamic_graph):
    pyg_list = []
    for G in dynamic_graph:
        x = torch.Tensor(G.graph["feature"])
        edge_index, edge_weight = torch_geometric.utils.from_scipy_sparse_matrix(nx.adjacency_matrix(G).astype(np.float32))
        pyg = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pyg_list.append(pyg)
    return pyg_list

def load_dataset(dataset_path):
    with open(dataset_path, "rb") as f:
        data, labels = pickle.load(f)
    return data, labels

def standardize_dataset(data, labels):
    dynamic_pygs = []
    for dynamic_graph in data:
        dynamic_pyg = dynamic_graph_to_pyg(dynamic_graph)
        dynamic_pygs.append(dynamic_pyg)

    labels_tensor = torch.Tensor(labels.reshape(labels.shape[0]))
    return dynamic_pygs, labels_tensor

