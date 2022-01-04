"""
Graph Neural Network Projects
Nick Kaparinos
2022
"""

from utilities import *
import torch
from os import makedirs
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import time

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f'Using device: {device}')
    print(f'{debugging = }')

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = f'logs/cora{time_stamp}/'
    makedirs(LOG_DIR, exist_ok=True)

    # Read Ezymes dataset
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    dataset = dataset.shuffle()
    num_node_features = dataset.num_node_features
    data = dataset[0].to(device)

    # GNN Model
    model = GNN_node_clasif_model(num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # output = model(fet)

    model.train()
    for epoch in range(2):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
