"""
Graph Neural Network Projects
Nick Kaparinos
2022
"""

from utilities import *
from os import makedirs
from torch_geometric.datasets import TUDataset
import time

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    dataset_name = 'PROTEINS'

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = f'logs/{dataset_name}_graphs_{time_stamp}/'
    makedirs(LOG_DIR, exist_ok=True)

    # Read dataset
    dataset = TUDataset(root='/tmp/TUDATASET', name=dataset_name, use_node_attr=True)
    dataset = dataset.shuffle()

    # Visualize
    n_graphs = 3
    for i in range(n_graphs):
        graph = dataset[i]
        visualise_graph(graph, visualisation_method='normal', save_figure=True, log_dir=LOG_DIR,
                        figure_name=f'{dataset_name}_graph{i}.png', title=f'{dataset_name} Dataset Graph {i}')

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
