"""
Graph Neural Network Projects
Nick Kaparinos
2022
"""

import random
import matplotlib.pyplot as plt
import optuna
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_add_pool
from tqdm import tqdm
import wandb

debugging = False


class GNN_node_clasif_model(torch.nn.Module):
    """ Pytorch GNN model for node classification """

    def __init__(self, num_node_features, num_classes, gnn_layer_type='GCN', n_gnn_layers=2, n_neurons=64,
                 n_linear_layers=1):
        super().__init__()
        if gnn_layer_type == 'GCN':
            gnn_layer = GCNConv
        elif gnn_layer_type == 'Graph_Sage':
            gnn_layer = SAGEConv
        elif gnn_layer_type == 'GIN':
            gnn_layer = GINConv
        else:
            raise ValueError('Unsupported GNN layer type!')

        if gnn_layer_type == 'GIN':
            self.conv1 = gnn_layer(torch.nn.Linear(num_node_features, n_neurons))
            self.gnn_layers = [gnn_layer(torch.nn.Linear(n_neurons, n_neurons)) for _ in range(n_gnn_layers - 1)]
            self.gnn_layers = torch.nn.ModuleList(self.gnn_layers)
        else:
            self.conv1 = gnn_layer(num_node_features, n_neurons)
            self.gnn_layers = [gnn_layer(n_neurons, n_neurons) for _ in range(n_gnn_layers - 1)]
            self.gnn_layers = torch.nn.ModuleList(self.gnn_layers)

        self.linear_layers = [torch.nn.Linear(n_neurons, n_neurons) for _ in range(n_linear_layers - 1)]
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)

        self.final_layer = torch.nn.Linear(n_neurons, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index)
            x = F.relu(x)

        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            x = F.relu(x)

        x = self.final_layer(x)
        return self.softmax(x)


class GNN_graph_clasif_model(torch.nn.Module):
    """ Pytorch GNN model for graph classification """

    def __init__(self, trial, num_node_features, num_classes, gnn_layer_type='GCN', n_gnn_layers=2, n_neurons=64,
                 n_linear_layers=1):
        super().__init__()
        if gnn_layer_type == 'GCN':
            gnn_layer = GCNConv
        elif gnn_layer_type == 'Graph_Sage':
            gnn_layer = SAGEConv
        elif gnn_layer_type == 'GIN':
            gnn_layer = GINConv
        else:
            raise ValueError('Unsupported GNN layer type!')

        if gnn_layer_type == 'GIN':
            # n_gin_linear_layers = trial.suggest_int('n_gin_linear_layers', 1, 3)
            n_gin_linear_layers = 1
            self.conv1 = gnn_layer(
                build_gin_mlp(n_gin_linear_layers, num_node_features, n_neurons, fist_conv_layer=True))

            self.gnn_layers = [gnn_layer(build_gin_mlp(n_gin_linear_layers, num_node_features, n_neurons)) for _ in
                               range(n_gnn_layers - 1)]
            self.gnn_layers = torch.nn.ModuleList(self.gnn_layers)
        else:
            self.conv1 = gnn_layer(num_node_features, n_neurons)
            self.gnn_layers = [gnn_layer(n_neurons, n_neurons) for _ in range(n_gnn_layers - 1)]
            self.gnn_layers = torch.nn.ModuleList(self.gnn_layers)

        self.linear_layers = [torch.nn.Linear(n_neurons, n_neurons) for _ in range(n_linear_layers - 1)]
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)

        self.final_layer = torch.nn.Linear(n_neurons, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index)
            x = F.relu(x)

        x = global_add_pool(x, batch)

        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            x = F.relu(x)

        x = self.final_layer(x)
        return self.softmax(x)


def GNN_node_hypermodel(trial, num_node_features, num_classes, device):
    """ Node classification GNN hypermodel """
    gnn_layer_type = trial.suggest_categorical('gnn_layer_type', ['GCN', 'Graph_Sage', 'GIN'])
    n_neurons = trial.suggest_int('n_neurons', 16, 256, 16)
    n_gnn_layers = trial.suggest_int('n_gnn_layers', 1, 3)
    n_linear_layers = trial.suggest_int('n_linear_layers', 1, 3)

    model = GNN_node_clasif_model(num_node_features, num_classes, gnn_layer_type, n_gnn_layers, n_neurons,
                                  n_linear_layers).to(device)
    name = f'{gnn_layer_type},neurons{n_neurons},gnn_layers{n_gnn_layers},linear_layers{n_linear_layers}'
    hyperparameters = {'gnn_layer_type': gnn_layer_type, 'n_neurons': n_neurons, 'n_gnn_layers': n_gnn_layers,
                       'n_linear_layers': n_linear_layers}
    return model, name, hyperparameters


def GNN_graph_hypermodel(trial, num_node_features, num_classes, device):
    """ Graph classification GNN hypermodel """
    gnn_layer_type = trial.suggest_categorical('gnn_layer_type', ['GCN', 'Graph_Sage', 'GIN'])
    n_neurons = trial.suggest_int('n_neurons', 16, 256, 16)
    n_gnn_layers = trial.suggest_int('n_gnn_layers', 1, 3)
    n_linear_layers = trial.suggest_int('n_linear_layers', 1, 3)

    model = GNN_graph_clasif_model(trial, num_node_features, num_classes, gnn_layer_type, n_gnn_layers, n_neurons,
                                   n_linear_layers).to(device)
    name = f'{gnn_layer_type},neurons{n_neurons},gnn_layers{n_gnn_layers},linear_layers{n_linear_layers}'
    hyperparameters = {'gnn_layer_type': gnn_layer_type, 'n_neurons': n_neurons, 'n_gnn_layers': n_gnn_layers,
                       'n_linear_layers': n_linear_layers}
    return model, name, hyperparameters


def define_objective(project, dataset, loss_fn, train_fn, hypermodel_fn, epochs, notes, seed, device):
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', low=1e-5, high=1e-1, step=0.001)
        batch_size = 32
        epoch_validation_accuracies = []

        # Model
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
        model, name, hyperparameters = hypermodel_fn(trial, num_node_features, num_classes, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        config = dict(hyperparameters,
                      **{'epochs': epochs, 'learning_rate': learning_rate, 'batch_size': batch_size, 'seed': seed})
        wandb.init(project=project, entity="nickkaparinos", name=name, config=config, notes=notes, group='',
                   reinit=True)

        for epoch in tqdm(range(1, epochs + 1)):
            validation_accuracy = train_fn(dataset, batch_size, model, loss_fn, optimizer, epoch, device)
            trial.report(validation_accuracy, epoch)
            epoch_validation_accuracies.append(validation_accuracy)

            # Pruning
            # if trial.should_prune():
            #     raise optuna.TrialPruned()

        max_validation_accuracy = max(epoch_validation_accuracies)
        wandb.log({'Max_validation_accuracy': max_validation_accuracy})
        return max_validation_accuracy

    return objective


def cora_train_fn(dataset, batch_size, model, loss_fn, optimizer, epoch, device) -> float:
    """ One epoch of Training and Validation using the Cora dataset """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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


def graph_clasif_train_fn(dataset, batch_size, model, loss_fn, optimizer, epoch, device) -> float:
    """ One epoch of Training and Validation a graph classification dataset """
    # Dataloaders
    indices = [i for i in range(len(dataset))]
    if debugging:
        dataset_train = torch.utils.data.Subset(dataset, indices[:10])
        dataset_val = torch.utils.data.Subset(dataset, indices[-10:])
    else:
        dataset_train = torch.utils.data.Subset(dataset, indices[:len(dataset) // 8])
        dataset_val = torch.utils.data.Subset(dataset, indices[len(dataset) // 8:])
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    y_train = np.empty((0,))
    y_train_pred = np.empty((0,))
    y_val = np.empty((0,))
    v_val_pred = np.empty((0,))

    # Training
    model.train()
    for batch_num, batch in enumerate(train_dataloader):
        # Inference
        batch = batch.to(device)
        output = model(batch)
        y_train_pred_temp = torch.argmax(output, dim=1).numpy()
        y_train_temp = batch.y

        #  Loss
        optimizer.zero_grad()
        loss = loss_fn(output, y_train_temp)
        loss.backward()
        optimizer.step()
        wandb.log(data={'Training_loss': loss.item()})

        # Stack
        y_train = np.hstack([y_train, y_train_temp.numpy()]) if y_train.size else y_train_temp.numpy()
        y_train_pred = np.hstack([y_train_pred, y_train_pred_temp]) if y_train_pred.size else y_train_pred_temp

    # Validation
    model.eval()
    with torch.no_grad():
        for batch_num, batch in enumerate(validation_dataloader):
            # Inference
            batch = batch.to(device)
            output = model(batch)
            y_val_pred_temp = torch.argmax(output, dim=1).numpy()
            y_val_temp = batch.y

            #  Loss
            val_loss = loss_fn(output, y_val_temp)
            wandb.log(data={'Validation_loss': val_loss.item()})

            # Stack
            y_val = np.hstack([y_val, y_val_temp.numpy()]) if y_val.size else y_val_temp.numpy()
            v_val_pred = np.hstack([v_val_pred, y_val_pred_temp]) if v_val_pred.size else y_val_pred_temp

    # Training and Validation Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='micro')
    validation_accuracy = accuracy_score(y_val, v_val_pred)
    validation_f1 = f1_score(y_val, v_val_pred, average='micro')

    # Wandb logging
    wandb.log(data={'Epoch': epoch, 'Training_accuracy': train_accuracy, 'Training_f1': train_f1,
                    'Validation_accuracy': validation_accuracy, 'Validation_f1': validation_f1})
    return validation_accuracy


def build_gin_mlp(n_gin_linear_layers, num_node_features, n_neurons, fist_conv_layer=False):
    """ Build mlp for GIN convolutional layer """
    layer_list = []
    if fist_conv_layer:
        layer_list.append(torch.nn.Linear(num_node_features, n_neurons))
    else:
        layer_list.append(torch.nn.Linear(n_neurons, n_neurons))

    for i in range(n_gin_linear_layers - 1):
        layer_list.append(torch.nn.ReLU())
        layer_list.append(torch.nn.Linear(n_neurons, n_neurons))

    return torch.nn.Sequential(*layer_list)


def visualise_graph(graph, visualisation_method='normal', save_figure=False, log_dir='/', figure_name='fig.png',
                    title='Graph', dpi=300):
    """ Visualize input graph """
    visualisation_fn_dict = {'normal': nx.draw_networkx, 'kamada_kawai': nx.draw_kamada_kawai,
                             'circular': nx.draw_circular}
    visualisation_fn = visualisation_fn_dict[visualisation_method]

    plt.figure(1)
    plt.clf()
    graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    visualisation_fn(graph)
    font = {'fontsize': 18}
    plt.title(title, **font)
    if save_figure:
        plt.savefig(log_dir + figure_name, dpi=dpi)
    else:
        plt.show()


def save_dict_to_file(dictionary, path, txt_name='hyperparameter_dict'):
    with open(f'{path}/{txt_name}.txt', 'w') as f:
        f.write(str(dictionary))


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
