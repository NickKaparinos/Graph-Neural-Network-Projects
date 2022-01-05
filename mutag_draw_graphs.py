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

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = f'logs/mutag_graphs_{time_stamp}/'
    makedirs(LOG_DIR, exist_ok=True)

    # Read MUTAG dataset
    dataset = TUDataset(root='/tmp/TUDATASET', name='MUTAG', use_node_attr=True)
    dataset = dataset.shuffle()

    # Visualize
    n_graphs = 3
    for i in range(n_graphs):
        graph = dataset[i]
        visualize_graph(graph, visualisation_method='normal', save_figure=True, log_dir=LOG_DIR,
                        figure_name=f'mutag_graph{i}.png', title=f'MUTAG Dataset Graph {i}')

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
