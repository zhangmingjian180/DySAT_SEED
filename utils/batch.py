import torch
from torch.utils.data import Dataset
from utils.preprocess import dynamic_graph_to_pyg

class MyDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dy_pyg = dynamic_graph_to_pyg(self.data[index])
        label = torch.Tensor(self.labels[index].reshape(1))
        return (dy_pyg, label)

    @staticmethod
    def collate_fn(samples):
        dynamic_graphs = [l[0] for l in samples]
        labels = [l[1] for l in samples]
        return (dynamic_graphs, labels)
