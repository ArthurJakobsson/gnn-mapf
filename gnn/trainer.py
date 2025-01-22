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
import subprocess
import pandas as pd

# import networkx as nx
import numpy as np

# from torch_geometric.datasets import TUDataset
# from dataloader import MyOwnDataset
from gnn.dataloader import MyOwnDataset
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter  # TODO: replace with Wandb
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import multiprocessing


# multiprocessing.set_start_method('spawn', force=True)

class GNNStack(nn.Module):
    def __init__(self, linear_dim, in_channels, hidden_dim, output_dim, edge_dim, relu_type, gnn_name, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.relu_type = relu_type
        self.gnn_func, self.use_edge_attr = self.gnn_func_from_name(gnn_name)
        
        self.convs = nn.ModuleList([self.build_image_conv_model(linear_dim, in_channels, hidden_dim)]) # image conv layer
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)])
        for _ in range(3):
            self.convs.append(self.build_graph_conv_model(hidden_dim, hidden_dim, edge_dim)) # graph conv layer
            self.lns.append(nn.LayerNorm(hidden_dim))
        self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
                                     nn.Linear(hidden_dim, output_dim))
        if task not in ['node', 'graph']:
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 4
    
    def gnn_func_from_name(self, gnn_name):
        # https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html
        # {gnn_name: (gnn_func, use_edge_attr), ...}
        gnn_dict = {"SAGEConv": (pyg_nn.SAGEConv, False),
                    "ResGatedGraphConv": (pyg_nn.ResGatedGraphConv, True),
                    }
        
        if gnn_name in gnn_dict:
            gnn_func, use_edge_attr = gnn_dict[gnn_name]
        else:
            raise ValueError(f"{gnn_name} not supported")
        
        print(f"use_edge_attr={use_edge_attr}")
        return gnn_func, use_edge_attr

    def build_image_conv_model(self, linear_dim, in_channels, hidden_dim):
        return CustomConv(linear_dim, in_channels, hidden_dim, self.relu_type, self.use_edge_attr)
    
    def build_graph_conv_model(self, in_channels, hidden_dim, edge_dim):
        if self.use_edge_attr:
            return self.gnn_func(in_channels, hidden_dim, edge_dim=edge_dim)
        else:
            return self.gnn_func(in_channels, hidden_dim) # not using edge_attr, so edge_dim=0

    def forward(self, data):
        """
        Input: data -- a torch_geometric.data.Data object with the following attributes:
            x -- node features
            edge_index -- graph connectivity
            batch -- batch assignment
            y -- node labels
        Output:
            F.log_softmax(x, dim=1) -- node score logits, we can do exp() to get probabilities
            """

        # priorities and edge features
        x = data.x  # (batch_size,2,D,D) float64
        edge_index = data.edge_index  # (2,num_edges)
        batch = data.batch  # (batch_size)
        bd_pred = data.bd_pred

        if self.use_edge_attr:
            edge_attr = data.edge_attr  # (num_edges,2+num_priority_copies)
            assert(edge_attr != None)
        else:
            edge_attr = None

        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)
        
        x = self.convs[0](x, bd_pred, edge_index, edge_attr)  # (batch_size, 
            
        for i in range(1, self.num_layers):
            temp = self.convs[i](x, edge_index, edge_attr)
            x = temp

            emb = x
            if self.relu_type!="relu":
                x = F.leaky_relu(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i != self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)
        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label, weights):
        base_loss = F.nll_loss(pred, label, reduction='none')
        weighted_loss = base_loss*weights
        # This combined with log_softmax in forward() is equivalent to cross entropy
        # as stated in https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        return weighted_loss.mean()

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, linear_dim, in_channels, out_channels, relu_type, use_edge_attr):
        super(CustomConv, self).__init__(aggr='add')
        self.lin = nn.Linear(linear_dim, out_channels)
        self.lin_self = nn.Linear(linear_dim, out_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.conv_self = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.relu_type=relu_type
        if relu_type == "relu":
            self.relu_func = F.relu 
        else:
            self.relu_func = F.leaky_relu
        self.use_edge_attr = use_edge_attr

    def forward(self, x, bd_pred, edge_index, edge_attr=None):
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)  # (2,num_edges)

        flattened_conv = torch.flatten(self.conv_self(x), start_dim=1)
        # TODO: use edge_attr

        if bd_pred.shape[0]>2:
            flattened_conv = torch.hstack([flattened_conv, bd_pred]) 

        # priorities
        # edge_attr has shape (num_edges,2+num_priority_copies)
        # flattened_conv = torch.hstack([flattened_conv, bd_pred])
            
        self_x = self.relu_func(flattened_conv)        
        self_x = self.lin_self(self_x)

        x_neighbors = self.relu_func(flattened_conv)
        x_neighbors = self.lin(x_neighbors)

        self_and_propagated = self_x + self.propagate(edge_index, x=x_neighbors)
        return self_and_propagated

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

def train(combined_dataset, writer, run_lr, edge_dim, relu_type, my_batch_size, dataset_size, gnn_name):

    # data_size = len(dataset)
    # loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    train_size = int(0.8 * len(combined_dataset))
    if dataset_size>0:
        train_size = dataset_size
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, test_size])
    BATCH_SIZE = my_batch_size #64 #1024
    NW = 4 # 32
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NW, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NW, pin_memory=True, persistent_workers=True)

    num_node_features = combined_dataset.datasets[0].num_node_features # datasets[0] is the first dataset in the combined dataset
    num_classes = combined_dataset.datasets[0].num_classes
    lin_dim = combined_dataset.datasets[0][0].lin_dim
    num_channels = combined_dataset.datasets[0][0].num_channels
    model = GNNStack(lin_dim,num_channels, 128, num_classes, edge_dim, relu_type=relu_type, gnn_name=gnn_name, task='node').to(device)
    opt = optim.AdamW(model.parameters(), lr=run_lr, weight_decay=5e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    min_loss = float('inf')
    max_test_acc = max_double_test_acc = float('-inf')

    patience = 10
    no_improvement = 0
    results = []

    for epoch in range(10):
        total_loss = 0
        correct = 0
        second_correct = 0
        total_samples = 0
        # start_time = time.time()
        
        model.train()
        for batch in tqdm(loader):
            batch = batch.to(device)
            opt.zero_grad()
            # with autocast():
            embedding, pred = model(batch)
            label = batch.y
            # pred = pred[batch.train_mask]
            # label = label[batch.train_mask]

            label_maxes = torch.argmax(label, dim=1).to(device)
            weights = torch.ones(label_maxes.shape).to(device) #batch.weights
            loss = model.loss(pred, label_maxes, weights)

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

        # runtime = time.time() - start_time
        
        # gpu_memory_usage = subprocess.check_output(
        #     ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        # ).decode("utf-8").strip()
        
        # test_acc, double_test_acc = test(test_loader, model)
        # results.append([my_batch_size, epoch, train_acc, test_acc, runtime, gpu_memory_usage])

        # print(f"Batchsize: {my_batch_size}, Epoch: {epoch}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Runtime: {runtime:.2f}s, GPU Usage: {gpu_memory_usage}MB")
        
        # if train_acc > 95:
        #     break

        test_acc, double_test_acc = test(test_loader, model)
        print(f"Epoch {epoch}. Loss: {total_loss:.4f}. Train accuracy: {train_acc:.2f}%. Train Top2 accuracy: {train_top2_acc:.2f}%. Test accuracy: {test_acc:.4f}. Top2 Test accuracy: {double_test_acc:.4f}")
        writer.add_scalar("test_accuracy", test_acc, epoch)
        writer.add_scalar("test_top2_accuracy", double_test_acc, epoch)

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
    # df = pd.DataFrame(results, columns=["Batchsize", "Epoch", "Train Acc", "Test Acc", "Runtime", "GPU Usage"])
    # df.to_csv(f"batch_experiment/batchsize_2workers_{my_batch_size}_stats.csv", index=False)

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
            # if not torch.all(first_choice == validation_pred): # Remove this as this can fail if two probs exactly equal
            #     print("WARNING: SOMETHING FISHY IN TEST AS FIRST CHOICE IS NOT EQUAL TO VALIDATION PREDICTION")
            label = data.y.argmax(dim=1)

            if model.task == 'node':
                # mask = data.val_mask if is_validation else data.test_mask
                # first_choice = first_choice[mask]
                # second_choice = second_choice[mask]
                label = data.y.argmax(dim=1)

            correct += first_choice.eq(label).sum().item()
            second_correct += second_choice.eq(label).sum().item()

            if model.task == 'graph':
                total += len(data.y)
            else:
                total += len(data.y) #mask.sum().item() if is_validation else torch.sum(data.test_mask).item()

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

### Example run
'''
python -m gnn.trainer --exp_folder=data_collection/data/logs/EXP_Test_batch --experiment=exp0 --iternum=0 --num_cores=4 \
  --processedFolders=data_collection/data/logs/EXP_Test_batch/iter0/processed \
  --k=5 \
  --m=3 \
  --num_priority_copies=10 \
  --lr=0.01 \
  --gnn_name=SAGEConv
'''
# python -m gnn.trainer --exp_folder=data_collection/data/logs/EXP_Test --experiment=exp0 --iternum=0 --num_cores=4 \
#   --processedFolders=data_collection/data/logs/EXP_Test3/iter0/processed \
#   --mapNpzFile=data_collection/data/benchmark_data/constant_npzs/all_maps.npz \
#   --bdNpzFolder=data_collection/data/benchmark_data/constant_npzs \
#   --pathNpzFolders=data_collection/data/logs/EXP_Test/iter0/eecbs_npzs

if __name__ == "__main__":
    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", help="experiment folder", type=str)
    parser.add_argument("--experiment", help="experiment name", type=str)
    parser.add_argument("--iternum", help="iteration name", type=int)
    parser.add_argument("--num_cores", help="num_cores", type=int)
    parser.add_argument("--k", help="window size", type=int)
    parser.add_argument("--m", help="num_nearby_agents", type=int)
    parser.add_argument("--num_priority_copies", help="copies of relative priority to include in input", type=int, default=1)
    parser.add_argument("--lr", help="learning_rate", type=float)
    parser.add_argument("--relu_type", help="relu type", type=str)
    parser.add_argument("--gnn_name", help="pytorch-geometric GNN to use", type=str, default="SAGEConv")
    # parser.add_argument("--mapNpzFile", help="map npz file", type=str, required=True)
    # parser.add_argument("--bdNpzFolder", help="bd npz file", type=str, required=True)
    parser.add_argument("--processedFolders", help="processed npz folders, comma seperated!", type=str, required=True)
    extraLayersHelp = "Types of additional layers for training, comma separated. Options are: agent_locations, agent_goal, at_goal_grid"
    parser.add_argument('--extra_layers', help=extraLayersHelp, type=str, default=None)
    parser.add_argument('--bd_pred', type=str, default=None, help="bd_predictions added to NN, type anything if adding")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset_size', type=int, default=-1)
    # parser.add_argument("--pathNpzFolders", help="path npz folders, comma seperated!", type=str, required=True)

    args = parser.parse_args()
    lr = args.lr
    edge_dim = 2 + args.num_priority_copies  # position difference between agents + repeated relative priority 
    relu_type = args.relu_type
    extra_layers = args.extra_layers
    bd_pred = args.bd_pred
    exp_folder, expname, iternum = args.exp_folder, args.experiment, args.iternum
    itername = "iter"+str(iternum)
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(f"./data_collection/data/logs/train_logs/"+expname+"_"+itername)
    if args.dataset_size>0:
        model_path = exp_folder+f"/{itername}"+f"/models_{args.dataset_size}/"
    else:
        model_path = exp_folder+f"/{itername}"+"/models/"
    finished_file = model_path + "/finished.txt"
    if os.path.exists(finished_file):
        print(f"Model already trained for {expname} {itername}")
        exit(0)

    os.makedirs(model_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ### Load the dataset
    # Each processed folder is a dataset
    dataset_list = []
    for folder in args.processedFolders.split(','):
        if not os.path.exists(folder):
            raise Exception(f"Folder {folder} does not exist!")
        dataset = MyOwnDataset(mapNpzFile=None, bdNpzFolder=None, pathNpzFolder=None,
                            processedOutputFolder=folder, num_cores=1, k=args.k, m=args.m, 
                            num_priority_copies=args.num_priority_copies, 
                            extra_layers=args.extra_layers, bd_pred=args.bd_pred, num_per_pt=16)
        dataset_list.append(dataset)

    # Combine into single large dataset
    dataset = torch.utils.data.ConcatDataset(dataset_list)
    print(f"Combined {len(dataset_list)} datasets for a combined size of {len(dataset)}")
    
    model = train(dataset, writer, lr, edge_dim, relu_type, args.batch_size, args.dataset_size, args.gnn_name)

    with open(f"{model_path}/finished.txt", "w") as f:
        f.write("")