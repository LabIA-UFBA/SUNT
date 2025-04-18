import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric_temporal.nn import recurrent as gnnr


class RecurrentA3TGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, periods):
        super(RecurrentA3TGCN, self).__init__()
        self.recurrent = gnnr.A3TGCN(in_channels, hidden_channels, periods)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class RecurrentDCRNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=1):
        super(RecurrentDCRNN, self).__init__()
        self.recurrent = gnnr.DCRNN(in_channels, hidden_channels, k)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class RecurrentTGCN(torch.nn.Module):
    '''
    TGCN model
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RecurrentTGCN, self).__init__()
        self.recurrent = gnnr.TGCN(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, prev_hidden_state):
        h = self.recurrent(x, edge_index, edge_weight, prev_hidden_state)
        y = F.elu(h) 
        y = self.linear(y)

        return y, h


class RecurrentGConvGRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=3):
        super(RecurrentGConvGRU, self).__init__()
        self.recurrent = gnnr.GConvGRU(in_channels, hidden_channels, k)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class RecurrentGConvLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=4):
        super(RecurrentGConvLSTM, self).__init__()
        self.recurrent = gnnr.GConvLSTM(in_channels, hidden_channels, k)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0


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


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor,  edge_weight: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        #x = self.conv1(x, edge_index, edge_weight)
        x = self.conv1(x, edge_index, edge_weight)
        x =  F.relu(x)
        x = self.linear(x) 
        
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


class GRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.gru = nn.GRU(in_channels, hidden_channels, batch_first=True)

        self.linear = nn.Linear(hidden_channels, out_channels)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x, _ = self.gru(x)
        x =  F.relu(x)
        x = self.linear(x)

        return x


class LSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True)

        self.linear = nn.Linear(hidden_channels, out_channels)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x, _ = self.lstm(x)
        x =  F.relu(x)
        x = self.linear(x)

        return x