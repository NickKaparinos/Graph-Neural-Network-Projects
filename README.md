# Graph Neural Network Projects
## Graph Neural Networks (GNNs)
A graph neural network (GNN) is a class of neural networks for processing data represented by graph data structures. They were popularized by their use in supervised learning on properties of various molecules.Since their inception, several variants of the simple message passing neural network (MPNN) framework have been proposed. Recently, Graph Neural Networks (GNNs) have gained increasing popularity in various domains, including social networks, knowledge graphs, life sciences and recommender systems. The power of GNNs in modeling the dependencies between nodes in a graph enables the breakthrough in the research area related to graph analysis. This repository contains various Graph Machine Learning projects solved using Deep Graph Neural Networks.

## Node Classification
Node classification is the supervised task, at which the labels of the nodes are predicted by the network. First, a Deep Graph Neural Network outputs an embedding for each node. Subsequently, the embeddings are passed through a Multi Layer Perceptron (MLP) head, which predicts the node labels.

### Cora Dataset
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. Nodes represent documents and edges represent citation links.

<p align="center"><img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/cora-graph" alt="drawing" width="500"/></p>

## Optimization Results
<p float="left">
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/contour.png" width="49.7%" /> 
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/optimization_history.png" width="49.7%" />
</p>

<p float="left">
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/param_importances.png" width="49.7%" /> 
  <img src="https://github.com/NickKaparinos/Graph-Neural-Network-Projects/blob/master/Images/cora/W%26B%20Chart%201_4_2022%2C%201_07_08%20PM.png" width="49.7%" />
</p>


## Optimal Model


## Graph CLassification
Graph classification is the supervised task, at which the label of the graph is predicted by the network. First, a Deep Graph Neural Network outputs an embedding for each node. Subsequently, the embeddings are pooled are pooled and passed through a Multi Layer Perceptron (MLP) head, which predicts the graph label.
