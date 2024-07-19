import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
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
# from dataloader import MyOwnDataset
from gnn.dataloader import MyOwnDataset
from torch_geometric.datasets import Planetoid
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import multiprocessing


multiprocessing.set_start_method('spawn', force=True)

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList([self.build_conv_model(input_dim, hidden_dim, True)])
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)])
        for _ in range(3):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, False))
            self.lns.append(nn.LayerNorm(hidden_dim))

        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
                                     nn.Linear(hidden_dim, output_dim))
        if task not in ['node', 'graph']:
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 4

    def build_conv_model(self, input_dim, hidden_dim, image_flag):
        if image_flag:
            return CustomConv(input_dim, hidden_dim)
        return pyg_nn.SAGEConv(input_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i != self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)
        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')
        linear_in = 98
        self.lin = nn.Linear(linear_in, out_channels)
        self.lin_self = nn.Linear(linear_in, out_channels)
        self.conv = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=1, padding=0)
        self.conv_self = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, x, edge_index):
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        self_x = F.relu(torch.flatten(self.conv_self(x), start_dim=1))
        self_x = self.lin_self(self_x)

        x_neighbors = F.relu(torch.flatten(self.conv(x), start_dim=1))
        x_neighbors = self.lin(x_neighbors)

        self_and_propogated = self_x + self.propagate(edge_index, x=x_neighbors)
        return self_and_propogated

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out

def save_models(model, total_loss, min_loss, test_acc, max_test_acc, double_test_acc, max_double_test_acc):
    if total_loss < min_loss:
        torch.save(model, model_path + '/min_train_loss.pt')
    if test_acc > max_test_acc:
        torch.save(model, model_path + '/max_test_acc.pt')
    if double_test_acc > max_double_test_acc:
        torch.save(model, model_path + '/max_double_test_acc.pt')

def train(dataset, task, writer):
    pdb.set_trace()
    data_size = len(dataset)
    loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True, num_workers=4, pin_memory=False)

    model = GNNStack(max(dataset.num_node_features, 1), 128, dataset.num_classes, task=task).to(device)
    opt = optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    min_loss = float('inf')
    max_test_acc = max_double_test_acc = float('-inf')

    patience = 10
    no_improvement = 0

    for epoch in tqdm(range(10)):
        total_loss = 0
        correct = 0
        second_correct = 0
        total_samples = 0

        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            with autocast():
                embedding, pred = model(batch)
                label = batch.y
                if task == 'node':
                    pred = pred[batch.train_mask]
                    label = label[batch.train_mask]

                label_maxes = torch.argmax(label, dim=1).to(device)
                loss = model.loss(pred, label_maxes)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * batch.num_graphs

            sorted_pred = torch.argsort(pred, axis=1)
            first_choice = sorted_pred[:, -1]
            second_choice = sorted_pred[:, -2]
            correct += first_choice.eq(label_maxes).sum().item()
            second_correct += second_choice.eq(label_maxes).sum().item()
            total_samples += label_maxes.size(0)

        total_loss /= len(loader.dataset)
        train_acc = correct / total_samples * 100
        train_top2_acc = (correct + second_correct) / total_samples * 100

        writer.add_scalar("loss", total_loss, epoch)
        writer.add_scalar("train_accuracy", train_acc, epoch)
        writer.add_scalar("train_top2_accuracy", train_top2_acc, epoch)
        scheduler.step(total_loss)

        if epoch % 5 == 0:
            test_acc, double_test_acc = test(test_loader, model)
            print(f"Epoch {epoch}. Loss: {total_loss:.4f}. Train accuracy: {train_acc:.2f}%. Train Top2 accuracy: {train_top2_acc:.2f}%. Test accuracy: {test_acc:.4f}. Top2 Test accuracy: {double_test_acc:.4f}")
            writer.add_scalar("test_accuracy", test_acc, epoch)
            writer.add_scalar("test_top2_accuracy", double_test_acc, epoch)

            if epoch >= 2:
                save_models(model, total_loss, min_loss, test_acc, max_test_acc, double_test_acc, max_double_test_acc)

            # # Early stopping logic
            # if total_loss < min_loss:
            #     min_loss = total_loss
            #     no_improvement = 0
            # else:
            #     no_improvement += 1

            # if no_improvement >= patience:
            #     print(f"Early stopping at epoch {epoch}")
            #     break

    return model

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    second_correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            emb, pred = model(data)
            sorted_pred = torch.argsort(pred, axis=1)
            first_choice = sorted_pred[:, -1]
            second_choice = sorted_pred[:, -2]
            validation_pred = pred.argmax(dim=1)
            assert(torch.all(first_choice == validation_pred))
            label = data.y.argmax(dim=1)

            if model.task == 'node':
                mask = data.val_mask if is_validation else data.test_mask
                first_choice = first_choice[mask]
                second_choice = second_choice[mask]
                label = data.y[mask].argmax(dim=1)

            correct += first_choice.eq(label).sum().item()
            second_correct += second_choice.eq(label).sum().item()

            if model.task == 'graph':
                total += len(data.y)
            else:
                total += mask.sum().item() if is_validation else torch.sum(data.test_mask).item()

    return correct / total, (correct + second_correct) / total

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

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# python -m gnn.trainer --exp_folder=data_collection/data/logs/EXP_Test --experiment=exp0 --iternum=0 --num_cores=4 --generate_initial=True
#   --mapNpzFile=data_collection/data/benchmark_data/constant_npzs/all_maps.npz
#   --bdNpzFolder=data_collection/data/benchmark_data/constant_npzs
#   --pathNpzFolders=data_collection/data/logs/EXP_Test/iter0/eecbs_npzs
if __name__ == "__main__":
    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", help="experiment folder", type=str)
    parser.add_argument("--experiment", help="experiment name", type=str)
    parser.add_argument("--iternum", help="iteration name", type=int)
    parser.add_argument("--num_cores", help="num_cores", type=int)
    parser.add_argument('--generate_initial', type=lambda x: bool(str2bool(x)))

    parser.add_argument("--mapNpzFile", help="map npz file", type=str)
    parser.add_argument("--bdNpzFolder", help="bd npz file", type=str)
    parser.add_argument("--pathNpzFolders", help="path npz folders, comma seperated!", type=str)

    args = parser.parse_args()
    exp_folder, expname, iternum = args.exp_folder, args.experiment, args.iternum
    itername = "iter"+str(iternum)
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(f"./data_collection/data/logs/train_logs/"+expname+"_"+itername)
    model_path = exp_folder+f"/{itername}"+"/models/"
    os.makedirs(model_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # device = "cpu"
    # print('Current cuda device: ',torch.cuda.get_device_name(0))

    

    dataset = MyOwnDataset(root=None, device=device, exp_folder=exp_folder, iternum=iternum, 
                        mapNpzFile=args.mapNpzFile, bdNpzFolder=args.bdNpzFolder, pathNpzFolders=args.pathNpzFolders.split(','),
                           num_cores=args.num_cores, generate_initial=args.generate_initial)
    dataset = dataset.shuffle()
    task = 'node'

    model = train(dataset, task, writer)
