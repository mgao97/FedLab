import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import community
import os
from dhg import Graph, Hypergraph
from dhg.data import Cooking200
from dhg.models import GCN

# Set the NUMEXPR_MAX_THREADS environment variable
os.environ["NUMEXPR_MAX_THREADS"] = "8"  # Replace 64 with the desired number of threads

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

# Define a custom Client class to hold data and model for each client
class Client:
    def __init__(self, data, model):
        self.data = data
        self.model = model

class MyGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Function to convert a list of node indices to a boolean tensor
def nodes_to_mask(node_indices, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[node_indices] = True
    return mask


# Function to perform local training on each client
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

    # Compute accuracy of the client model on the local test dataset
    client_model.eval()
    with torch.no_grad():
        test_output = client_model(data)
        test_pred = test_output[test_mask].argmax(dim=1)
        test_accuracy = (test_pred == y[test_mask]).sum().item() / len(y[test_mask])

    return test_accuracy


# Function for model evaluation on a test dataset
def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index, y = data.x, data.edge_index, data.y
        output = model(data)
        pred = output.argmax(dim=1)
        accuracy = (pred == y).sum().item() / len(y)
    return accuracy


# Main function to execute local learning (non-federated approach) for GCN
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
    # print(graph_data)

    # Partition the graph data into subgraphs using the Louvain algorithm
    data_partitions = torch.load('best_partitions.pt')
    num_clients = len(data_partitions)

    # Create client objects with data partitions
    clients_data = []
    for client_data in data_partitions:
        client_graph_data = graph_data.subgraph(nodes_to_mask(client_data, graph_data.num_nodes))

        # initialize the GCN model for each client (the same model architecture)
        client_model = MyGCN(
            input_dim=client_graph_data.x.shape[1],
            hidden_dim=32,
            output_dim=len(torch.unique(client_graph_data.y))
        )

        # create a Client object with the subgraph data and the client model
        client = Client(data=client_graph_data, model=client_model)

        # Append the Client object to the list of clients
        clients_data.append(client)

    # Train each client locally (without federated learning)
    client_accuracy = []
    for client in clients_data:
        test_accuracy = train_client_model(
            client_model=client.model,
            data=client.data,
            num_epochs=200,
            learning_rate=0.001
        )
        print("Client Accuracy: {:.2f}%".format(test_accuracy * 100))
        client_accuracy.append(test_accuracy)
    avg_client_accuracy = sum(client_accuracy) / len(client_accuracy)
    print("Average Test Accuracy of All Clients: {:.2f}%".format(avg_client_accuracy * 100))
    # No need for model aggregation, as we are not using federated learning in this case

    # Use the trained client models for evaluation
    # client_accuracy = []
    # for client in clients_data:
    #     test_accuracy = evaluate_model(client.model, client.data)
    #     print("Test Accuracy of Client: {:.2f}%".format(test_accuracy * 100))
    #     client_accuracy.append(test_accuracy)
    #     # Compute the average test accuracy of all clients
    # avg_client_accuracy = sum(client_accuracy) / len(client_accuracy)
    # print("Average Test Accuracy of All Clients: {:.2f}%".format(avg_client_accuracy * 100))


if __name__ == "__main__":
    main()
