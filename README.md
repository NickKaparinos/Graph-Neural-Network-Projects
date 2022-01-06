# Graph Neural Network Projects

A graph neural network (GNN) is a class of neural networks for processing data represented by graph data structures. They were popularized by their use in supervised learning on properties of various molecules.Since their inception, several variants of the simple message passing neural network (MPNN) framework have been proposed. Recently, Graph Neural Networks (GNNs) have gained increasing popularity in various domains, including social networks, knowledge graphs, life sciences and recommender systems. The power of GNNs in modeling the dependencies between nodes in a graph enables the breakthrough in the research area related to graph analysis. This repository contains various Graph Machine Learning projects solved using Deep Graph Neural Networks.

--------------------------------------------------------------------------------

- [Node Classification](#node-classification)
  * [Cora Dataset](#cora-dataset)
- [Graph Classification](#graph-classification)
  * [MUTAG Dataset](#mutag-dataset)
  * [PROTEINS Dataset](#proteins-dataset)

# Node Classification
Node classification is the supervised task, at which the labels of the nodes are predicted by the network. First, a Deep Graph Neural Network outputs an embedding for each node. Subsequently, the embeddings are passed through a Multi Layer Perceptron (MLP) head, which predicts the node labels.

## Cora Dataset
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. Nodes represent documents and edges represent citation links.

<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/cora-graph" alt="drawing" width="500"/></p>

### Optimization Results
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/contour.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/W%26B%20Chart%201_4_2022%2C%201_07_08%20PM.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/param_importances.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/optimization_history.png" alt="drawing"/></p>

### Optimal Model

<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/cora_optimal_learning_curve.png" alt="drawing"/></p>

The optimal model achieved **84%** validation accuracy.

# Graph Classification
Graph classification is the supervised task, at which the label of the graph is predicted by the network. First, a Deep Graph Neural Network outputs an embedding for each node. Subsequently, the embeddings are pooled and passed through a Multi Layer Perceptron (MLP) head, which predicts the graph label.

## MUTAG Dataset
The MUTAG dataset consists of 188 chemical compounds divided into two classes according to their mutagenic effect on a bacterium. The chemical data was obtained from http://cdb.ics.uci.edu and converted to graphs, where vertices represent atoms and edges represent chemical bonds. Explicit hydrogen atoms have been removed and vertices are labeled by atom type and edges by bond type (single, double, triple or aromatic).
<p float="left">
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/mutag/mutag_graph0.png" width="32.9%" /> 
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/mutag/mutag_graph1.png" width="32.9%" />
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/mutag/mutag_graph2.png" width="32.9%" />
</p>

### Optimization Results
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/mutag/contour.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/mutag/mutag_parallel_coordinates_plot.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/mutag/param_importances.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/mutag/optimization_history.png" alt="drawing"/></p>

### Optimal Model
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/mutag/mutag_optimal_model_curve.png" alt="drawing"/></p>


The optimal model achieved **83.6%** validation accuracy.

## PROTEINS Dataset
PROTEINS is a dataset of proteins that are classified as enzymes or non-enzymes. Nodes represent the amino acids and two nodes are connected by an edge if they are less than 6 Angstroms apart. It consists of 1113 graphs with 39.06 nodes and 72.82 edges per graph on average.
<p float="left">
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/proteins/PROTEINS_graph0.png" width="32.9%" /> 
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/proteins/PROTEINS_graph1.png" width="32.9%" />
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/proteins/PROTEINS_graph2.png" width="32.9%" />
</p>

### Optimization Results
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/proteins/contour.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/proteins/proteins_parallel_coordinates_plot.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/proteins/param_importances.png" alt="drawing"/></p>
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/proteins/optimization_history.png" alt="drawing"/></p>

### Optimal Model
<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/proteins/proteins_learning_curve.png" alt="drawing"/></p>

The optimal model achieved **74.1%** validation accuracy.
