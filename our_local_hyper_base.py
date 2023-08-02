import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import community
import os
from copy import deepcopy
from dhg import Graph, Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN

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

# def find_subhypergraph_with_nodes(hypergraph, node_list):
#     # Create a new hypergraph with the specified nodes
#     subgraph_nodes = set(node_list)
#     subgraph_edges = []

#     for edge in hypergraph.e:
#         # Include the edge in the subgraph if any of its nodes are in the node list
#         if any(node in subgraph_nodes for node in edge):
#             subgraph_edges.append(edge)
    
#     # Handle isolated nodes
#     isolated_nodes = [node for node in subgraph_nodes if not any(node in edge for edge in subgraph_edges)]
#     for node in isolated_nodes:
#         # Connect isolated node to all other hyperedges
#         subgraph_edges.append((node,))

#     subgraph_hg = Hypergraph(len(subgraph_nodes), subgraph_edges)

#     return subgraph_hg

def find_subhypergraph_with_nodes(hypergraph, node_list):
    # Create a new hypergraph with the specified nodes
    subgraph_nodes = set(node_list)
    subgraph_edges = set()

    for edge in hypergraph.e:
        # Include the edge in the subgraph if any of its nodes are in the node list
        if any(node in subgraph_nodes for node in edge):
            subgraph_edges.add(tuple(sorted(edge)))  # Sorting ensures uniqueness

    # Handle isolated nodes
    isolated_nodes = {node for node in subgraph_nodes if not any(node in edge for edge in subgraph_edges)}
    for node in isolated_nodes:
        # Connect isolated node to all other hyperedges
        subgraph_edges.add((node,))
        # subgraph_nodes.add(node)  # Do not include isolated nodes in subgraph_nodes to keep the count correct

    # The number of nodes in the subhypergraph should be equal to the number of unique nodes in the hyperedges
    subgraph_nodes = set(node for edge in subgraph_edges for node in edge)

    subgraph_hg = Hypergraph(len(subgraph_nodes), list(subgraph_edges))

    return subgraph_hg

def remap_masks(original_mask, client_nodes_indices):
    """
    Remaps the original mask to the client's local subgraph.

    Args:
        original_mask (torch.BoolTensor): The original mask (e.g., train_mask or test_mask) of the entire graph.
        client_nodes_indices (list or torch.Tensor): List or tensor containing the indices of nodes in the client's local subgraph.

    Returns:
        torch.BoolTensor: The remapped mask corresponding to the client's local subgraph.
    """
    # Create a boolean mask with False for all nodes in the entire graph
    remapped_mask = torch.zeros_like(original_mask, dtype=torch.bool)

    # Set True for nodes that belong to the client's local subgraph
    remapped_mask[client_nodes_indices] = True

    return remapped_mask



# Define a custom Client class to hold data and model for each client
class Client:
    def __init__(self, data, model, X, lbl, train_mask, val_mask, test_mask):
        self.data = data
        self.model = model
        self.X = X
        self.lbl = lbl
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

# class MyGCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MyGCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)


# Function to convert a list of node indices to a boolean tensor
def nodes_to_mask(node_indices, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[node_indices] = True
    return mask


# Function to perform local training on each client
def train_client_model(client_model, X, lbl, train_mask, val_mask, test_mask, data, num_epochs, learning_rate):

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val = 0
    for epoch in range(num_epochs):
        # Prepare data and labels for the client
        optimizer.zero_grad()
        output = client_model(X[train_mask], data)
        loss = F.cross_entropy(output[train_mask], lbl[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            with torch.no_grad():
                val_output = client_model(X, data)[val_mask]
                val_pred = val_output.argmax(dim=1)
                val_accuracy = (val_pred == lbl[val_mask]).sum().item() / len(lbl[val_mask])
                if val_accuracy > best_val:
                    best_val = val_accuracy
                    best_state = deepcopy(client_model.state_dict())
                    torch.save(best_state,'local_hgnn.model')
                    print(f'epoch: {epoch}, loss: {loss.item():.5f}, val_acc: {val_accuracy}')

    # Compute accuracy of the client model on the local test dataset
    client_model.load_state_dict(best_state)
    test_output = client_model(X, data)
    test_pred = test_output[test_mask].argmax(dim=1)
    test_accuracy = (test_pred == lbl[test_mask]).sum().item() / len(lbl[test_mask])

    return test_accuracy


# # Function for model evaluation on a test dataset
# def evaluate_model(model, data):
#     model.eval()
#     with torch.no_grad():
#         x, edge_index, y = data.x, data.edge_index, data.y
#         output = model(data)
#         pred = output.argmax(dim=1)
#         accuracy = (pred == y).sum().item() / len(y)
#     return accuracy


# Main function to execute local learning (non-federated approach) for GCN
def main():
    # load data
    hdataset = Cooking200()
    X, lbl = torch.eye(hdataset["num_vertices"]), hdataset["labels"]
    HG = Hypergraph(hdataset['num_vertices'], hdataset['edge_list'])


    # G = Graph.from_hypergraph_clique(HG, weighted=True)
    # node0, node1, edge_weight = [], [], []
    # for edge in G.e[0]:
    #     node0.append(edge[0])
    #     node1.append(edge[1])
    # for weight in G.e[1]:
    #     edge_weight.append(weight)
    # edge_index = torch.tensor([node0, node1],dtype=int)
    # edge_weight = torch.tensor(edge_weight,dtype=float)
    train_mask, val_mask, test_mask = hdataset['train_mask'], hdataset['val_mask'], hdataset['test_mask']

    
    print(f'train_mask {train_mask}, shape of train_mask {train_mask.shape}')

    # Partition the graph data into subgraphs using the Louvain algorithm
    data_partitions = torch.load('best_partitions.pt')
    num_clients = len(data_partitions)

    # Create client objects with data partitions
    clients_data = []
    for client_data in data_partitions:
        client_graph_data = find_subhypergraph_with_nodes(HG, client_data)
        print('=======================')
        print(client_graph_data)

        # initialize the HGNN model for each client (the same model architecture)
        client_model = HGNN(
            in_channels=torch.eye(client_graph_data.num_v).shape[1],
            hid_channels=32,
            num_classes=20,
            use_bn=True,
        )

        # Create a new 'Client' object with remapped masks
        client = Client(
            data=client_graph_data,
            X=torch.eye(client_graph_data.num_v),  # Assuming X is the feature matrix for the client's local subgraph
            lbl=lbl[torch.tensor(client_data)],  # Assuming lbl is the label tensor for the entire graph, and we use the train_mask
            train_mask = remap_masks(train_mask, client_data),
            val_mask = remap_masks(val_mask, client_data),
            test_mask = remap_masks(test_mask, client_data),
            model=client_model
        )
        # Append the Client object to the list of clients
        clients_data.append(client)

    # Train each client locally (without federated learning)
    client_accuracy = []
    for client in clients_data:
        test_accuracy = train_client_model(
            client_model=client.model,
            X=client.X,
            lbl=client.lbl,
            train_mask=client.train_mask,
            val_mask=client.val_mask,
            test_mask=client.test_mask,
            data=client.data,
            num_epochs=200,
            learning_rate=0.01
        )
        print("Client Accuracy: {:.2f}%".format(test_accuracy * 100))
    avg_client_accuracy = sum(client_accuracy) / len(client_accuracy)
    print("Average Test Accuracy of All Clients: {:.2f}%".format(avg_client_accuracy * 100))
    # No need for model aggregation, as we are not using federated learning in this case

    # # Use the trained client models for evaluation
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
