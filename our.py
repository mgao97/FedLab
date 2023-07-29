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


# Function to perform local training on each client
def train_client_model(client_model, client_data, num_epochs, learning_rate):
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for data, target in client_data:
            # Prepare data and labels for the client
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()




def louvain_partition(graph, num_partitions):
    # Convert graph to networkx graph
    nx_graph = nx.from_edgelist(graph.edge_index.t().tolist())

    # Apply the Louvain algorithm for community detection
    partition = community.best_partition(nx_graph, resolution=1.0)

    # Create a list of nodes in each partition
    partitioned_data = [[] for _ in range(num_partitions)]
    for node, part_id in partition.items():
        partitioned_data[part_id].append(node)

    return partitioned_data


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
def federated_train(server_model, clients_data, num_epochs, learning_rate):
    for epoch in range(num_epochs):
        for client in clients_data:
            # Train the client model on its data
            train_client_model(
                client_model=client.model,
                client_data=client.train_data,
                num_epochs=num_epochs,
                learning_rate=learning_rate
            )

        # Aggregate client models to update the server model
        aggregate_models(server_model, [client.model for client in clients_data])

    return server_model

# Main function to execute federated learning for GCN
def main():
    # load data
    print('*********************')
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

    print(graph_data)

    # # Initialize GCN model and server model
    # gcn_model = MyGCN(...)  # Replace with your GCN model initialization
    # server_model = MyGCN(...)  # Replace with your GCN model initialization

    # # Partition the graph data into subgraphs and assign them to clients using FedLab's functionalities
    # # Assume you have a list of data samples and labels (data_list and label_list)
    # num_clients = 10
    # data_partitions = louvain_partition(graph_data, num_clients)

    # # Create client objects with data partitions
    # clients_data = [Client(data=client_data, model=gcn_model) for client_data in data_partitions]

    # # Start federated training
    # global_model = federated_train(
    #     server_model=server_model,
    #     clients_data=clients_data,
    #     num_epochs=10,
    #     learning_rate=0.01
    # )

    # # Use the trained global model for predictions or further evaluations

if __name__ == "__main__":
    main()
