import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime
from tqdm import tqdm
import os

import numpy as np

from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from dataloader import MyOwnDataset
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
        for l in range(3):  # increase the number of layers
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, False))
            self.lns.append(nn.LayerNorm(hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 4  # increase the number of layers

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
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):  # currently, both == 1
        super(CustomConv, self).__init__(aggr='add')  # "sum" aggregation.
        linear_in = 98  # TODO calculate the number of output pixels, using a formula and not hardcoded
        self.lin = nn.Linear(linear_in, out_channels)
        self.lin_self = nn.Linear(linear_in, out_channels)
        conv_channels_in = 2
        conv_channels_out = 2
        self.conv = nn.Conv2d(conv_channels_in, conv_channels_out, kernel_size=(3, 3), stride=1, padding=0)
        self.conv_self = nn.Conv2d(conv_channels_in, conv_channels_out, kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, x, edge_index):
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        self_x = torch.flatten(self.conv_self(x), start_dim=1)
        self_x = F.relu(self_x)
        self_x = self.lin_self(self_x)

        x_neighbors = torch.flatten(self.conv(x), start_dim=1)
        x_neighbors = F.relu(x_neighbors)
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
    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GNNStack(max(dataset.num_node_features, 1), 128, dataset.num_classes, task=task)
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    min_loss = float('inf')
    max_test_acc = max_double_test_acc = float('-inf')

    patience = 10
    no_improvement = 0

    for epoch in tqdm(range(100)):
        total_loss = 0
        correct = 0
        second_correct = 0
        total_samples = 0  # Track total number of samples for accuracy computation

        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]

            label_maxes = torch.zeros(label.shape[0]).type(torch.LongTensor).to(device)
            for i in range(len(label)):
                label_maxes[i] = torch.argmax(label[i])
            loss = model.loss(pred, label_maxes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            total_loss += loss.item() * batch.num_graphs

            sorted_pred = torch.argsort(pred, axis=1)
            first_choice = sorted_pred[:, -1]
            second_choice = sorted_pred[:, -2]
            correct += first_choice.eq(label_maxes).sum().item()
            second_correct += second_choice.eq(label_maxes).sum().item()
            total_samples += label_maxes.size(0)  # Increment total samples by the batch size

        total_loss /= len(loader.dataset)
        train_acc = correct / total_samples * 100  # Correct train accuracy as percentage
        train_top2_acc = (correct + second_correct) / total_samples * 100  # Top-2 train accuracy as percentage

        writer.add_scalar("loss", total_loss, epoch)
        writer.add_scalar("train accuracy", train_acc, epoch)
        writer.add_scalar("train top2 accuracy", train_top2_acc, epoch)
        scheduler.step(total_loss)

        if epoch % 5 == 0:
            test_acc, double_test_acc = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Train accuracy: {:.2f}%. Train Top2 accuracy: {:.2f}%. Test accuracy: {:.4f}. Top2 Test accuracy: {:.4f}".format(
                epoch, total_loss, train_acc, train_top2_acc, test_acc, double_test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)
            writer.add_scalar("top2 test accuracy", double_test_acc, epoch)

            if epoch >= 50:
                save_models(model, total_loss, min_loss, test_acc, max_test_acc, double_test_acc, max_double_test_acc)

            # if total_loss < min_loss:
            #     min_loss = total_loss
            #     no_improvement = 0
            # else:
            #     no_improvement += 1

            # if no_improvement >= patience:
            #     print("Early stopping at epoch {}".format(epoch))
            #     break

    return model


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    second_correct = 0
    for data in loader:
        data = data.to(device)

        with torch.no_grad():
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
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
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

if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter("./log/" + current_time)
    model_path = "./model_log/" + current_time
    os.mkdir(model_path)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print('Current CUDA device:', torch.cuda.get_device_name(0))
        else:
            print('CUDA is not available. Using CPU.')
    except Exception as e:
        print("An error occurred while initializing CUDA:", e)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dataset = MyOwnDataset(root='data', device=device)
    dataset = dataset.shuffle()
    task = 'node'

    model = train(dataset, task, writer)
