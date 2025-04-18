import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn


class GATEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GATEdgeClassifier, self).__init__()
        self.conv1 = gnn.GATConv(in_channels, hidden_channels, heads)
        self.conv2 = gnn.GATConv(hidden_channels * heads, hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, w):
        # Node embeddings
        x = self.conv1(x, edge_index, w)
        x = F.relu(x)
        x = self.conv2(x, edge_index, w)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)


class SAGEEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGEEdgeClassifier, self).__init__()
        self.conv1 = gnn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = gnn.SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)


class GCNEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEdgeClassifier, self).__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, w):
        # Node embeddings
        x = self.conv1(x, edge_index, w)
        x = F.relu(x)
        x = self.conv2(x, edge_index, w)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        return self.classifier(edge_embeddings)


class ChebEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, filter_size=3):
        super(ChebEdgeClassifier, self).__init__()
        self.conv1 = gnn.ChebConv(in_channels, hidden_channels, K=filter_size)
        self.conv2 = gnn.ChebConv(hidden_channels, hidden_channels, K=filter_size)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, w):
        # Node embeddings
        x = self.conv1(x, edge_index, w)
        x = F.relu(x)
        x = self.conv2(x, edge_index, w)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)


class LEConvGNNEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LEConvGNNEdgeClassifier, self).__init__()
        self.conv1 = gnn.LEConv(in_channels, hidden_channels)
        self.conv2 = gnn.LEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, w):
        # Node embeddings
        x = self.conv1(x, edge_index, w)
        x = F.relu(x)
        x = self.conv2(x, edge_index, w)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)
    

class SSGConvEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SSGConvEdgeClassifier, self).__init__()
        self.conv1 = gnn.SSGConv(in_channels, hidden_channels, 0.1)
        self.conv2 = gnn.SSGConv(hidden_channels, hidden_channels, 0.1)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, w):
        # Node embeddings
        x = self.conv1(x, edge_index, w)
        x = F.relu(x)
        x = self.conv2(x, edge_index, w)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)



class EGConvEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EGConvEdgeClassifier, self).__init__()
        self.conv1 = gnn.EGConv(in_channels, hidden_channels)
        self.conv2 = gnn.EGConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)



class AntiSymmetricConvEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(AntiSymmetricConvEdgeClassifier, self).__init__()
        self.conv1 = gnn.AntiSymmetricConv(in_channels)
        self.conv2 = gnn.AntiSymmetricConv(in_channels)
        self.classifier = nn.Linear(2 * in_channels, out_channels)

    def forward(self, x, edge_index):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)


# -----------------------------------------------------------------------

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


# -------------------------------------------------------------------------------



class SuperGATConvEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(SuperGATConvEdgeClassifier, self).__init__()
        self.conv1 = gnn.SuperGATConv(in_channels, hidden_channels, heads, dropout=0.2)
        self.conv2 = gnn.SuperGATConv(hidden_channels, hidden_channels, heads, dropout=0.2)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)
    


class PANConvEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=1):
        super(PANConvEdgeClassifier, self).__init__()
        self.conv1 = gnn.PANConv(in_channels, hidden_channels, filter_size=k)
        self.conv2 =  gnn.PANConv(hidden_channels, hidden_channels, filter_size=k)
        self.classifier = nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Node embeddings
        x, _ = self.conv1(x, edge_index)
        x = F.relu(x)
        x, _ = self.conv2(x, edge_index)
        
        # Edge embeddings
        row, col = edge_index
        edge_embeddings = torch.cat([x[row], x[col]], dim=-1)
        
        # Edge classification
        return self.classifier(edge_embeddings)