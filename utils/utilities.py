def to_device(graph_l, label_l, device):
    graph_l = [[G.to(device) for G in g ] for g in graph_l]
    label_l = [label.to(device) for label in label_l]
    return (graph_l, label_l)
