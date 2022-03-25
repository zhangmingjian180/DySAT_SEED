import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.preprocess import load_dataset, standardize_dataset
from utils.batch import  MyDataset
from utils.utilities import to_device
from models.model import DySAT

from sklearn.model_selection import train_test_split

def get_accuracy(model, dataset_test, device):
    dataloader = DataLoader(dataset_test,
                            batch_size=1,
                            shuffle=True,
                            collate_fn=dataset_test.collate_fn)

    model.eval()
    count = 0
    for graphs_l, label_l in dataloader:
        graphs_l, label_l = to_device(graphs_l, label_l, device)
        result = model(graphs_l[0])
        if abs(result - label_l[0]) < 0.5:
            count += 1

    return count / len(dataloader)

def train(args, model, dataset_train, dataset_test, device):
    dataloader = DataLoader(dataset_train,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=dataset_train.collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_func = nn.BCELoss()

    # in training
    best_test = 0
    patient = 0
    for epoch in range(args.epochs):
        model.train()
        for graphs_l, label_l in dataloader:
            graphs_l, label_l = to_device(graphs_l, label_l, device)
            opt.zero_grad()
            result = model(graphs_l[0])
            loss = loss_func(result, label_l[0])
            loss.backward()
            opt.step()
            print(loss.item())

        acc = get_accuracy(model, dataset_test, device)
        print("Epoch {}, Test AUC = {}".format(epoch, acc))

        if acc > best_test:
            best_test = acc
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.patient:
                break

    model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    acc = get_accuracy(model, dataset_test, device)
    print("Best Test AUC = {}".format(acc))
    acc = get_accuracy(model, dataset_train, device)
    print("Best Train AUC = {}".format(acc))


def test(model, dataset_test, device):
    acc = get_accuracy(model, dataset_test, device)
    print("Test AUC = {}".format(acc))

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, default=40, help="the length of dynamic graph")
    parser.add_argument('--node_length', type=int, default=10, help="the length of node")
    
    # Experimental settings.
    parser.add_argument('--dataset_path', default="./data/data_and_labels.pkl", help='dataset path')
    parser.add_argument('--device', default="cpu", help='the device to use')
    parser.add_argument('--epochs', type=int, default=1, help='# epochs')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, default=1, help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (# nodes)')
    parser.add_argument("--patient", type=int, default=3, help="patient")
    parser.add_argument('--pretrained', type=bool, default="", help='whether import param')
    parser.add_argument('--test', type=bool, default="", help='whether test')
    
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--residual', type=bool, default="True", help='whether use residual') 
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate for self-attention model.')
    parser.add_argument('--spatial_drop', type=float, default=0.1, help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, default=0.5, help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Initial learning rate for self-attention model.')
    
    # Architecture params
    parser.add_argument('--structural_head_config', default='16,8,8', help='Encoder layer config: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_config', default='128', help='Encoder layer config: # units in each GAT layer')
    parser.add_argument('--temporal_head_config', default='16', help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', default='128', help='Encoder layer config: # units in each Temporal layer')
    parser.add_argument('--window', type=int, default=-1, help='Window for temporal attention (default : -1 => full)')
    
    args = parser.parse_args()
    print(args)

    graphs_train, graphs_test, labels_train, labels_test = train_test_split(*load_dataset(args.dataset_path), test_size=0.1)

    dataset_train = MyDataset(graphs_train, labels_train)
    dataset_test = MyDataset(graphs_test, labels_test)
    device = torch.device(args.device)

    model = DySAT(args, args.node_length, args.time_steps).to(device)

    if args.pretrained:
        model.load_state_dict(torch.load("./model_checkpoints/model.pt"))

    if args.test:
        test(model, dataset_test, device)
    else:
        train(args, model, dataset_train, dataset_test, device)










