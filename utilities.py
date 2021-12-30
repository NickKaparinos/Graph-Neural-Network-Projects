"""
Graph Neural Network Projects
Nick Kaparinos
2022
"""

import numpy as np
import random
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import wandb
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

debugging = False


class GCN_model(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_node_features)
        # self.linear = torch.nn.Linear(num_node_features, 128)
        self.linear = torch.nn.LazyLinear(out_features=num_classes)

        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        return self.softmax(x)


def visualize_graph(g):
    g = torch_geometric.utils.to_networkx(g, to_undirected=True)
    # nx.draw(g)
    nx.draw_networkx()
    # nx.draw_kamada_kawai(g)
    # nx.draw_circular(g)

    plt.show()


def visualize_cora_graph(g):
    g = torch_geometric.utils.to_networkx(g, to_undirected=True)
    nx.draw_networkx(g)
    # nx.draw_kamada_kawai(g, node_size=10)
    # nx.draw_circular(g)
    plt.show()


def cora_train_one_epoch(dataloader, model, loss_fn, optimizer, epoch, device) -> float:
    """ One epoch of Training and Validation using the Cora dataset """
    model.train()
    for batch_num, batch in enumerate(dataloader):
        # Masks
        train_mask = batch.train_mask + batch.val_mask
        val_mask = batch.test_mask
        if debugging:
            train_mask = torch.zeros(train_mask.shape[0], dtype=torch.bool)

        # Inference
        batch = batch.to(device)
        output = model(batch)
        training_output = output[train_mask]
        validation_output = output[val_mask]

        training_predictions = torch.argmax(training_output, dim=1).tolist()
        validation_predictions = torch.argmax(validation_output, dim=1).tolist()

        training_labels = batch.y[train_mask]
        validation_labels = batch.y[val_mask].tolist()

        #  Loss
        optimizer.zero_grad()
        loss = loss_fn(training_output, training_labels)
        loss.backward()
        optimizer.step()

        # Training and Validation Metrics
        training_labels = training_labels.tolist()
        train_accuracy = accuracy_score(training_labels, training_predictions)
        train_f1 = f1_score(training_labels, training_predictions, average='micro')

        validation_accuracy = accuracy_score(validation_labels, validation_predictions)
        validation_f1 = f1_score(validation_labels, validation_predictions, average='micro')

        # Wandb logging
        wandb.log(data={'Epoch': epoch, 'Training_loss': loss.item(), 'Training_accuracy': train_accuracy,
                        'Training_f1': train_f1, 'Validation_accuracy': validation_accuracy,
                        'Validation_f1': validation_f1})
    return validation_accuracy


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
