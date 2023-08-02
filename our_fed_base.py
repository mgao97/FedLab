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
import matplotlib.pyplot as plt
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
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(MyGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
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
    def __init__(self, data, model, num_samples):
        self.data = data
        self.model = model
        self.num_samples = num_samples  # Number of data samples in the client


# Function to convert a list of node indices to a boolean tensor
def nodes_to_mask(node_indices, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[node_indices] = True
    return mask

def compute_class_weights(labels):
    # labels: tensor containing the class labels (e.g., data.y in your case)

    # Count the occurrences of each class label
    class_counts = torch.bincount(labels)

    # Compute the class weights as the reciprocal of class frequencies
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)

    # Normalize the class weights to sum up to 1
    class_weights /= class_weights.sum()

    return class_weights


# Function to perform local training on each client
def train_client_model(client_model, data, num_epochs, learning_rate, dropout_prob=0.5):
    # Extract the required data from the 'data' object

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # Define the loss function with class weights
    class_weights = compute_class_weights(data.y)
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    

    # Lists to store training loss and validation accuracy for each epoch
    train_loss_list = []
    best_val_acc = 0
    best_model_state_dict = None

    for epoch in range(num_epochs):
        client_model.train()
        # Prepare data and labels for the client
        optimizer.zero_grad()
        output = client_model(data)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # calculate and print the training loss for this epoch
        train_loss_list.append(loss.item())
        

        # calculate and print the validation accuracy for this epoch
        val_accuracy = evaluate_model(client_model, data)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            # Save the model's state dictionary when it achieves the best validation accuracy
            best_model_state_dict = client_model.state_dict()
            torch.save(best_model_state_dict, 'fed_gcn_base.model')
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f} Validation Accuracy: {val_accuracy:.2f}%")

    # plot the training loss and validation accuracy
    plt.figure(figsize=(5,5))
    
    plt.plot(range(1, num_epochs+1), train_loss_list, label='Client Fed GCN Model')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig('client_fed_gcn_train_loss.png')
    # plt.show()


# Function for model aggregation (e.g., Federated Averaging)
def aggregate_models(server_model, clients_data):
    server_model_dict = server_model.state_dict()
    total_samples = 0  # To keep track of the total number of data samples across clients

    # Initialize the server_model_dict as zeros
    for key in server_model_dict:
        server_model_dict[key] = torch.zeros_like(server_model_dict[key])

    # Aggregate the model parameters from all client models
    for client in clients_data:
        num_samples = client.num_samples  # Number of data samples in the client
        total_samples += num_samples

        client_model_dict = client.model.state_dict()
        for key in server_model_dict:
            server_model_dict[key] += client_model_dict[key] * num_samples

    # Calculate the weighted average of the parameters
    for key in server_model_dict:
        server_model_dict[key] /= total_samples

    # server_model.load_state_dict(server_model_dict)

    return server_model

# Main federated training function
def federated_train(server_model, data, clients_data, num_epochs, num_round, learning_rate):
    
    # Lists to store training loss and validation accuracy for each epoch of the server model
    server_train_loss_list = []
    best_server_val_acc = 0.0
    best_server_model_state_dict = None


    for epoch in range(num_round):
        for client in clients_data:
            # Train the client model on its data
            train_client_model(
                client_model=client.model,
                data=client.data,  # Assuming the data is stored in client.data
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                
            )

        # Aggregate client models to update the server model
        server_model = aggregate_models(server_model, clients_data)
        

        # Calculate and print the training loss for this epoch of the server model
        server_train_loss = torch.nn.CrossEntropyLoss()(server_model(data)[data.train_mask], data.y[data.train_mask])
        server_train_loss_list.append(server_train_loss.item())
        

        # Calculate and print the validation accuracy for this epoch of the server model
        server_val_accuracy = evaluate_model(server_model, data)

        if server_val_accuracy > best_server_val_acc:
            best_server_val_acc = server_val_accuracy
            # Save the model's state dictionary when it achieves the best validation accuracy
            best_server_model_state_dict = deepcopy(server_model.state_dict())

            print(f"Epoch [{epoch+1}/{num_round}] Server Training Loss: {server_train_loss.item():.4f} Server Validation Accuracy: {server_val_accuracy:.2f}%")
        

    # Plot the training loss and validation accuracy for the server model
    plt.figure(figsize=(5, 5))
    
    plt.plot(range(1, num_round + 1), server_train_loss_list, label="Server Fed GCN Model")
    plt.xlabel("Epoch")
    plt.ylabel("Server Training Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig('server_fed_gcn_train_loss.png')
    # plt.show()

    server_model.load_state_dict(best_server_model_state_dict)

    return server_model


# Function for model evaluation on a val/test dataset
def evaluate_model(model, data, test=False):
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        if test:
            correct = pred[data.test_mask] == data.y[data.test_mask]
        else:
            correct = pred[data.val_mask] == data.y[data.val_mask]

        accuracy = correct.sum().item() / len(correct)
    return accuracy

# Main function to execute federated learning for GCN
def main():
    # load data
    
    hdataset = Cooking200()
    X, lbl = torch.eye(hdataset["num_vertices"]), hdataset["labels"]

    unique_values, counts = torch.unique(lbl, return_counts=True)
    print("Unique Values:", unique_values)
    print("Counts:", counts)

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
    class_weights = compute_class_weights(lbl)
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
        client = Client(data=client_graph_data, model=client_model, num_samples=client_graph_data.y.shape[0])

        # Append the Client object to the list of clients
        clients_data.append(client)
    
    
    # initialize the server model
    server_model = MyGCN(input_dim=graph_data.x.shape[1], hidden_dim=32, output_dim=len(torch.unique(graph_data.y)))  # Replace with your GCN model initialization
    

    # Start federated training
    global_model = federated_train(
        server_model=server_model,
        data = graph_data,
        clients_data=clients_data,
        num_epochs=100,
        num_round=30,
        learning_rate=0.01
    )
    
    # Evaluate the trained global model on the test dataset
    test_accuracy = evaluate_model(global_model, graph_data, test=True)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))


if __name__ == "__main__":
    main()
