import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime
import pdb
from tqdm import tqdm
import os
import argparse

import networkx as nx
import numpy as np

from torch_geometric.datasets import TUDataset
from dataloader import MyOwnDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim, True))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, False))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim, image_flag):
        if image_flag:
            return CustomConv(input_dim,hidden_dim)
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels): # currently, both == 1
        super(CustomConv, self).__init__(aggr='add')  # "sum" aggregation.
        linear_in = 98 # TODO calculate the number of output pixels, using a formula and not hardcoded
        self.lin = nn.Linear(linear_in, out_channels)
        self.lin_self = nn.Linear(linear_in, out_channels)
        conv_channels_in = 2
        conv_channels_out = 2
        self.conv = nn.Conv2d(conv_channels_in, conv_channels_out, kernel_size=(3,3), stride=1, padding=0)
        self.conv_self = nn.Conv2d(conv_channels_in, conv_channels_out, kernel_size=(3,3), stride=1, padding=0)
        '''
        architecture questions
        - should we pad? ("most relevant info should be near center") - no need rn
        - should we make it 2x9x9? (same logic) - yes
        - should we have multiple layers of convolution? - good for now
        '''

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix
        # edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Remove self-edges
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        self_x = torch.flatten(self.conv_self(x), start_dim=1)
        self_x = F.relu(self_x)
        self_x = self.lin_self(self_x)

        # For self edges (see tutorial on gnn pyg)
        x_neighbors = torch.flatten(self.conv(x), start_dim=1)
        x_neighbors = F.relu(x_neighbors)
        x_neighbors = self.lin(x_neighbors)

        self_and_propogated = self_x + self.propagate(edge_index, x=x_neighbors)

        # For removed self-edges
        return  self_and_propogated# TODO is everything ok without size arg? size=(x.size(0), x.size(0))

        # For self edges
        # return self.propagate(edge_index, size=(x.size(0),x.size(0)),x=x)

    def message(self, x_j, edge_index, size):
        # Compute messages
        # x_j has shape [E, out_channels]
        # pdb.set_trace()
        # row, col = edge_index
        # deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)+1
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # # if any(torch.isnan(norm.view(-1,1) * x_j).flatten()):
        #     # pdb.set_trace()
        # old_norm = norm.view(-1,1) * x_j
        # return old_norm
        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out

def save_models(model, total_loss,min_loss, test_acc, max_test_acc, double_test_acc, max_double_test_acc):
    if total_loss<min_loss:
        torch.save(model, model_path+'/min_train_loss.pt')
    if test_acc>max_test_acc:
        torch.save(model, model_path+'/max_test_acc.pt')
    if double_test_acc>max_double_test_acc:
        torch.save(model, model_path+'/max_double_test_acc.pt')

def train(dataset, task, writer):
    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)
        # pdb.set_trace()

    # build model
    model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    min_loss = float('inf')
    max_test_acc = max_double_test_acc = float('-inf')

    # train
    for epoch in tqdm(range(1000)):
        total_loss = 0
        model.train()
        for batch in loader:
            #print(batch.train_mask, '----')
            batch = batch.to(device)
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            # lab_1_hot = torch.zeros(pred.shape[0]).type(torch.LongTensor)
            label_maxes = torch.zeros(label.shape[0]).type(torch.LongTensor).to(device)
            for i in range(len(label)):
            #     lab_1_hot[i] = torch.argmax(pred[i])
                label_maxes[i] = torch.argmax(label[i])
            loss = model.loss(pred, label_maxes)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 5 == 0:
            test_acc, double_test_acc = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}. Top2 Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc, double_test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)
            writer.add_scalar("top2 test accuracy", double_test_acc, epoch)

            if epoch >= 50:
                save_models(model, total_loss,min_loss, test_acc, max_test_acc, double_test_acc, max_double_test_acc)

    return model

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    second_correct = 0
    for data in loader:
        data = data.to(device)
        pdb.set_trace()
        with torch.no_grad():
            emb, pred = model(data)
            #change top two using argsort
            sorted_pred = torch.argsort(pred, axis=1)
            first_choice = sorted_pred[:,-1]
            second_choice = sorted_pred[:,-2]
            validation_pred = pred.argmax(dim=1)
            assert(torch.all(first_choice == validation_pred))
            label = data.y.argmax(dim=1)

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            # pred = pred[mask]
            first_choice = first_choice[mask]
            second_choice = second_choice[mask]
            label = data.y[mask].argmax(dim=1)

        correct += first_choice.eq(label).sum().item() # number first correct
        second_correct += second_choice.eq(label).sum().item() # number second correct

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total, (correct+second_correct)/total

def visualize():
  color_list = ["red", "orange", "green", "blue", "purple", "brown"]

  loader = DataLoader(dataset, batch_size=64, shuffle=True)
  embs = []
  colors = []
  for batch in loader:
      emb, pred = model(batch)
      embs.append(emb)
      colors += [color_list[y] for y in batch.y]
  embs = torch.cat(embs, dim=0)

  xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
  plt.scatter(xs, ys, color=colors)


if __name__ == "__main__":
    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="experiment folder", type=str)
    parser.add_argument("experiment", help="experiment name", type=str)
    parser.add_argument("iternum", help="iteration name", type=int)
    args = parser.parse_args()

    folder, expname, iternum = args.folder, args.experiment, args.iternum

    itername = "iter"+iternum

    writer = SummaryWriter(f"../data_collection/data/logs/train_logs"+expname+"_"+itername)
    model_path = folder+"/models/"
    os.mkdir(model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print('Current cuda device: ',torch.cuda.get_device_name(0))

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dataset = MyOwnDataset(root=f"{folder}/labels/", device=device, iternum=iternum)
    dataset = dataset.shuffle()
    task = 'node'

    model = train(dataset, task, writer)
