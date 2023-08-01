import torch
import torch.nn.functional as F
# from federatedscope.core.models import MyGCN  # Replace this with your GCN model class
# from fedlab.core.client import Client
from fedlab.core.client.trainer import ClientTrainer

# from fedlab.core.server.handler import FedAvgParameterServer
from fedlab.core.server.manager import AsynchronousServerManager
# from fedlab.utils.functional import partition_data
# from fedlab.utils.dataset.sampling import Partitioner
# from fedlab.utils.functional import batch_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import numpy as np
import copy
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Graph, Hypergraph
from dhg.data import Cooking200
from dhg.models import GCN

from torch_geometric.data import Data
import torch
import networkx as nx
import community  # louvain community detection

import os

# Set the NUMEXPR_MAX_THREADS environment variable
os.environ["NUMEXPR_MAX_THREADS"] = "8"  # Replace 64 with the desired number of threads



class MyGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

import community  # louvain community detection
import torch
import networkx as nx

def louvain_partition(graph_data):
    # Convert graph to networkx graph
    nx_graph = nx.from_edgelist(graph_data.edge_index.t().tolist())

    # Apply the Louvain algorithm for community detection
    partition = community.best_partition(nx_graph, resolution=1.0)

    # Create a list of nodes in each partition
    num_partitions = max(partition.values()) + 1
    partitioned_data = [[] for _ in range(num_partitions)]
    for node, part_id in partition.items():
        partitioned_data[part_id].append(node)

    return partitioned_data

# def find_best_part(graph_data, data_partitions):
#         best_part = []
#         for part in data_partitions:
#             best_part.append(len(part))
#         if best_part == [1767, 1303, 743, 1340, 1106, 1144]:
#             torch.save(data_partitions, 'best_partitions.pt')
#         else:
#             data_partitions = louvain_partition(graph_data)
#             find_best_part(graph_data, data_partitions)
#         return data_partitions
def find_best_part(graph_data, data_partitions):
    # Check if the current partitions contain exactly 20 classes of node labels
    num_classes_per_partition = [len(torch.unique(graph_data.y[nodes])) for nodes in data_partitions]
    if num_classes_per_partition == [20, 20, 20, 20, 20, 20]:
        torch.save(data_partitions, 'best_partitions.pt')
        return data_partitions
    else:
        # If the current partitions do not satisfy the condition, apply Louvain community detection again
        data_partitions = louvain_partition(graph_data)
        return find_best_part(graph_data, data_partitions)


# Define a custom Client class to hold data and model for each client
class Client:
    def __init__(self, data, model):
        self.data = data
        self.model = model

# Function to convert a list of node indices to a boolean tensor
def nodes_to_mask(node_indices, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[node_indices] = True
    return mask

# Function to perform local training on each client
# def train_client_model(client_model, data, num_epochs, learning_rate):
#     # Extract the required data from the 'data' object
#     x, edge_index, edge_weight, y, train_mask, val_mask, test_mask = (
#         data.x, data.edge_index, data.edge_weight, data.y, data.train_mask,
#         data.val_mask, data.test_mask
#     )

#     # Define optimizer and loss function
#     optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)
#     criterion = torch.nn.CrossEntropyLoss()

#     for epoch in range(num_epochs):
#         # Prepare data and labels for the client
#         optimizer.zero_grad()
#         output = client_model(data)
#         print(output.shape, y.shape)
#         loss = criterion(output[train_mask], y[train_mask])
#         loss.backward()
#         optimizer.step()
def train_client_model(client_model, data, num_epochs, learning_rate):
    # Extract the required data from the 'data' object
    x, edge_index, edge_weight, y, train_mask, val_mask, test_mask = (
        data.x, data.edge_index, data.edge_weight, data.y, data.train_mask,
        data.val_mask, data.test_mask
    )

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Prepare data and labels for the client
        optimizer.zero_grad()
        output = client_model(data)
        loss = criterion(output[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

# Function for model aggregation (e.g., Federated Averaging)
def aggregate_models(server_model, client_models):
    server_model_dict = server_model.state_dict()
    for client_model in client_models:
        client_model_dict = client_model.state_dict()
        for key in server_model_dict:
            server_model_dict[key] += client_model_dict[key]
    num_clients = len(client_models)
    for key in server_model_dict:
        server_model_dict[key] /= num_clients
    server_model.load_state_dict(server_model_dict)

# Main federated training function
def federated_train(server_model, clients_data, num_epochs, num_round, learning_rate):
    for epoch in range(num_round):
        for client in clients_data:
            # Train the client model on its data
            train_client_model(
                client_model=client.model,
                data=client.data,  # Assuming the data is stored in client.data
                num_epochs=num_epochs,
                learning_rate=learning_rate
            )

        # Aggregate client models to update the server model
        aggregate_models(server_model, [client.model for client in clients_data])

    return server_model


# Function for model evaluation on a test dataset
def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index, y = data.x, data.edge_index, data.y
        output = model(data)
        pred = output.argmax(dim=1)
        accuracy = (pred[data.test_mask] == y[data.test_mask]).sum().item() / len(y[data.test_mask])
    return accuracy

# Main function to execute federated learning for GCN
def main():
    # load data
    
    hdataset = Cooking200()
    X, lbl = torch.eye(hdataset["num_vertices"]), hdataset["labels"]
    HG = Hypergraph(hdataset['num_vertices'], hdataset['edge_list'])
    G = Graph.from_hypergraph_clique(HG, weighted=True)
    node0, node1, edge_weight = [], [], []
    for edge in G.e[0]:
        node0.append(edge[0])
        node1.append(edge[1])
    for weight in G.e[1]:
        edge_weight.append(weight)
    edge_index = torch.tensor([node0, node1],dtype=int)
    edge_weight = torch.tensor(edge_weight,dtype=float)
    train_mask, val_mask, test_mask = hdataset['train_mask'], hdataset['val_mask'], hdataset['test_mask']
    graph_data = Data(x=X, edge_index= edge_index, edge_wight=edge_weight, y=lbl, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    print(lbl, lbl.shape, torch.unique(lbl))
    print(f'graph_data {graph_data}')

    # Partition the graph data into subgraphs and assign them to clients using FedLab's functionalities
    # Assume you have a list of data samples and labels (data_list and label_list)
    
    # data_partitions = louvain_partition(graph_data)

    # data_partitions = find_best_part(graph_data, data_partitions)

    
    data_partitions = torch.load('best_partitions.pt')
    num_clients = len(data_partitions)
    

    # Create client objects with data partitions
    clients_data = []
    for client_data in data_partitions:
        print('=========================')
        print(len(client_data))
        client_graph_data = graph_data.subgraph(nodes_to_mask(client_data, graph_data.num_nodes))

        print('*********************')
        print(client_graph_data)

        # initialize the GCN model for each client (the same model architecture)
        client_model =MyGCN(
            input_dim=client_graph_data.x.shape[1],
            hidden_dim=32,
            output_dim=len(torch.unique(client_graph_data.y))
        )

        # create a Client object with the subgraph data and the client model
        client = Client(data=client_graph_data, model=client_model)

        # Append the Client object to the list of clients
        clients_data.append(client)
    
    # initialize the server model
    server_model = MyGCN(input_dim=graph_data.x.shape[1], hidden_dim=32, output_dim=len(torch.unique(graph_data.y)))  # Replace with your GCN model initialization
    

    # Start federated training
    global_model = federated_train(
        server_model=server_model,
        clients_data=clients_data,
        num_epochs=200,
        num_round=100,
        learning_rate=0.01
    )

    # Use the trained global model for predictions or further evaluations
    # Evaluate the trained global model on the test dataset
    test_accuracy = evaluate_model(global_model, graph_data)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))


if __name__ == "__main__":
    main()
