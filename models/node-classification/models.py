import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric import nn as gnn

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x) 
        #x = F.log_softmax(x, dim=1)
        
        return x      
   

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = gnn.GATConv(in_channels, hidden_channels, heads)
        self.linear = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = self.linear(x)
        
        return x


class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.SAGEConv(in_channels, hidden_channels)
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.linear_out(x)

        return x

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, filter_size=3):
        super().__init__()
        self.conv1 = gnn.ChebConv(in_channels, hidden_channels, K=filter_size)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor,  edge_weight: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x)

        return x


class GATv2ConvNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = gnn.GATv2Conv(in_channels, hidden_channels, heads)
        self.linear = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = self.linear(x)
        
        return x


class AntiSymmetricConvGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        self.conv1 = gnn.AntiSymmetricConv(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.linear(x)
        
        return x
    

class EGConvGNN(torch.nn.Module):
    '''
    The Efficient Graph Convolution from the 
    "Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions" paper.
    '''
    def __init__(self, in_channels, hidden_channels, out_channels,):
        super().__init__()
        self.conv1 = gnn.EGConv(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.linear(x)
        
        return x
    

class GPSConvNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mp, heads):
        super().__init__()
        self.conv1 = gnn.GPSConv(in_channels, mp, heads, dropout=0.2)
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.linear(x)
        
        return x
    

class SSGConvNN(torch.nn.Module): # ok
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.SSGConv(in_channels, hidden_channels, 0.1)
        
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x) 
        #x = F.log_softmax(x, dim=1)
        
        return x   
    

class _FAConvGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.FAConv(in_channels, out_channels)
        
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x) 
        #x = F.log_softmax(x, dim=1)
        
        return x   
    

class FAConvGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.FAConv(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, x_0: Tensor,  edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, x_0, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x) 
        #x = F.log_softmax(x, dim=1)
        
        return x   



class LEConvGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.LEConv(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        
        x = self.conv1(x, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x) 
        
        return x  


class SuperGATConvNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = gnn.SuperGATConv(in_channels, hidden_channels, heads, dropout=0.2)
        self.linear = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.linear(x)
        
        return x


class GATv2ConvNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, h=1):
        super().__init__()
        self.conv1 = gnn.GATv2Conv(in_channels, hidden_channels, heads=h, dropout=0.2)
        self.linear = nn.Linear(hidden_channels * h, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = self.linear(x)
        
        return x


class PANConvNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=1):
        super().__init__()
        self.conv1 = gnn.PANConv(in_channels, hidden_channels, filter_size=k)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x, _ = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.linear(x)
        
        return x