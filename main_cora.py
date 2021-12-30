"""
Graph Neural Network Projects
Nick Kaparinos
2022
"""

from utilities import *
import torch
import torch_geometric
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
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

    # Read enzymes dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    dataset = dataset.shuffle()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes

    # Wandb
    name = 'test'
    config = ''
    notes = ''
    wandb.init(project='gnn-projects', entity="nickkaparinos", name=name, config=config, notes=notes,
               group='test', reinit=True)

    # GNN Model
    model = GCN_model(num_node_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    # TODO add optuna

    # Training
    epochs = 20
    loss_fn = torch.nn.NLLLoss()
    for epoch in tqdm(range(1, epochs + 1)):
        cora_train_one_epoch(dataloader, model, loss_fn, optimizer, epoch, device)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
