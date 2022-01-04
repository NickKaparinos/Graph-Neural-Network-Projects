"""
Graph Neural Network Projects
Nick Kaparinos
2022
"""

from utilities import *
import torch
from os import makedirs
import optuna
import logging
import sys
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from pickle import dump
import time

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f'{debugging = }')

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = f'logs/cora{time_stamp}/'
    makedirs(LOG_DIR, exist_ok=True)

    # Read Cora dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    dataset = dataset.shuffle()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes

    # Hyperparameter optimisation
    project = 'Cora-GNN'
    study_name = f'cora_study_{time_stamp}'
    model_type = 'node_classification'
    epochs = 12
    loss_fn = torch.nn.NLLLoss()
    notes = ''
    objective = define_objective(project=project, dataset=dataset, loss_fn=loss_fn, train_fn=cora_train_one_epoch,
                                 hypermodel_fn=GNN_node_hypermodel, epochs=epochs, model_type=model_type, notes=notes,
                                 device=device)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed), study_name=study_name,
                                direction='maximize', pruner=optuna.pruners.HyperbandPruner(),
                                storage=f'sqlite:///{LOG_DIR}{study_name}.db', load_if_exists=True)
    study.optimize(objective, n_trials=None, timeout=2*60)
    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best value: {study.best_value}')

    # Save results
    results_dict = {'Best_hyperparameters': study.best_params, 'Best_value': study.best_value, 'study_name': study_name,
                    'log_dir': LOG_DIR}
    save_dict_to_file(results_dict, LOG_DIR, txt_name='study_results')
    study.trials_dataframe().to_csv(LOG_DIR + "study_results.csv")

    # Plot study results
    plots = [(optuna.visualization.plot_optimization_history, "optimization_history.png"),
             (optuna.visualization.plot_intermediate_values, "intermediate_values.png"),
             (optuna.visualization.plot_parallel_coordinate, "parallel_coordinate.png"),
             (optuna.visualization.plot_contour, "contour.png"),
             (optuna.visualization.plot_param_importances, "param_importances.png")]
    figs = []
    for plot_function, plot_name in plots:
        fig = plot_function(study)
        figs.append(fig)
        fig.update_layout(title=dict(font=dict(size=20)), font=dict(size=15))
        fig.write_image(LOG_DIR + plot_name, width=1920, height=1080)
    with open(LOG_DIR + 'result_figures.pkl', 'wb') as f:
        dump(figs, f)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
