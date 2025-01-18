import numpy as np
import scipy as sp
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn import GCNConv

from Utils import normalize_adj


class IDKN_simple(nn.Module):
    def __init__(self, in_channels=10, hidden_channels=64, out_channels=1):
        super(IDKN_simple, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x,x2, edge_index, num_nodes):
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 输出层
        x = self.lin(x)
        return x.squeeze()

# Embedding Dimension 1-32-16-8
# Attention Head      8-4-2
class IDKN(torch.nn.Module):
    def __init__(self):
        super(IDKN, self).__init__()

        self.conv1 = GATConv( 10, 4, heads=8, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv2 = GATConv( 32, 4, heads=4, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv3 = GATConv( 16, 4, heads=2, concat=True, negative_slope=0.2, dropout=0.2)

        self.lin1 = Linear(10, 1, bias=True)
        self.lin2 = Linear(8, 1, bias=False)

        self.activation = nn.ReLU()

    def forward(self, x1, x2, edge_index, num_nodes):
        fill_value = 1
        Adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  sparse_sizes=(num_nodes, num_nodes))
        Adj_matrix = fill_diag(Adj_matrix, fill_value)

        Adj = F.normalize(Adj_matrix.to_dense(), p=1, dim=1)

        # Feature Scoring
        #init_score = self.lin1(x1)
        #init_score = self.activation(init_score)

        # Encoding Representation
        x3 = self.conv1(x1, edge_index)
        x3 = self.activation(x3)
        x4 = self.conv2(x3, edge_index)
        x4 = self.activation(x4)
        x5 = self.conv3(x4, edge_index)
        x5 = self.activation(x5)
        x6 = F.dropout(x5, p=0.3, training=self.training)

        # Local Scoring
        R = torch.matmul(x6, x6.t())
        R_1 = torch.mul(R, Adj)
        R_2 = torch.sum(R_1, dim=1, keepdim=True)
        normalied_degree = x1[:, 0].view(-1, 1)
        local_score = R_2 + normalied_degree
        #local_score = R_2

        # Global Socring
        R_3 = torch.matmul(Adj, x6)
        global_score = self.lin2(R_3)

        ranking_scores = torch.add(local_score, global_score)
        return ranking_scores



class IDKN_cat(torch.nn.Module):
    def __init__(self):
        super(IDKN_cat, self).__init__()

        # GATConv layers
        self.conv1 = GATConv(20, 4, heads=8, concat=True, negative_slope=0.2, dropout=0.1)  # 8 = new input dimension
        self.conv2 = GATConv(32, 4, heads=4, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv3 = GATConv(16, 4, heads=2, concat=True, negative_slope=0.2, dropout=0.2)

        # Linear layers
        self.lin1 = Linear(10 + 10, 8, bias=True)  # Input dimension updated to include role vector (e.g., 7 + 3)
        self.lin2 = Linear(8, 1, bias=False)

        self.activation = nn.ReLU()

    def forward(self, x1, x2, edge_index, num_nodes):
        """
        x: 中心性向量
        x2: 角色向量
        edge_index: 边索引
        num_nodes: 节点数量
        """
        fill_value = 1
        Adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  sparse_sizes=(num_nodes, num_nodes))
        Adj_matrix = fill_diag(Adj_matrix, fill_value)
        Adj = F.normalize(Adj_matrix.to_dense(), p=1, dim=1)

        # Feature Fusion (Centerality + Role)
        x_combined = torch.cat([x1, x2], dim=1)  # Concatenate features along the feature dimension

        # Feature Scoring
        #init_score = self.lin1(x_combined)

        # Encoding Representation
        x3 = self.conv1(x_combined, edge_index)
        x4 = self.conv2(x3, edge_index)
        x5 = self.conv3(x4, edge_index)
        x6 = F.dropout(x5, p=0.3, training=self.training)

        # Local Scoring
        R = torch.matmul(x6, x6.t())
        R_1 = torch.mul(R, Adj)
        R_2 = torch.sum(R_1, dim=1, keepdim=True)
        normalied_degree = x1[:, 0].view(-1, 1)
        local_score = R_2 + normalied_degree

        # Global Scoring
        R_3 = torch.matmul(Adj, x6)
        global_score = self.lin2(R_3)

        # Combine local and global scores
        ranking_scores = torch.add(local_score, global_score)
        return ranking_scores

class IDKN_Attention(nn.Module):
    def __init__(self):
        super(IDKN_Attention, self).__init__()

        # GATConv layers
        self.conv1 = GATConv(16, 4, heads=8, concat=True, negative_slope=0.2, dropout=0.1)  # 8 = cross-attention output dim
        self.conv2 = GATConv(32, 4, heads=4, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv3 = GATConv(16, 4, heads=2, concat=True, negative_slope=0.2, dropout=0.2)

        # Attention parameters
        self.query_linear = Linear(10, 16)  # Transform x1 (中心性特征)
        self.key_linear = Linear(10, 16)   # Transform x2 (角色特征)
        self.value_linear = Linear(20, 16)

        # Linear layers for scoring
        self.lin1 = Linear(16, 8, bias=True)
        self.lin2 = Linear(8, 1, bias=False)

    def forward(self, x1, x2, edge_index, num_nodes):
        """
        x: 中心性向量
        x2: 角色向量
        edge_index: 边索引
        num_nodes: 节点数量
        """
        # Step 1: Compute cross-attention weights
        query = self.query_linear(x1)  # Transform x1 (中心性特征)
        key = self.key_linear(x2)     # Transform x2 (角色特征)
        value = self.value_linear(torch.cat([x1, x2], dim=1))

        attention_weights = F.softmax(torch.bmm(query.unsqueeze(1), key.unsqueeze(2)).squeeze(), dim=-1)  # Cross-attention
        attention_weights = attention_weights.unsqueeze(-1)  # 变成 [batch_size, 1]

        fused_features = attention_weights * value

        # Step 2: GAT layers
        x1 = self.conv1(fused_features, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = F.dropout(x3, p=0.3, training=self.training)

        # Step 3: Local Scoring
        fill_value = 1
        Adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  sparse_sizes=(num_nodes, num_nodes))
        Adj_matrix = fill_diag(Adj_matrix, fill_value)
        Adj = F.normalize(Adj_matrix.to_dense(), p=1, dim=1)

        R = torch.matmul(x4, x4.t())
        R_1 = torch.mul(R, Adj)
        R_2 = torch.sum(R_1, dim=1, keepdim=True)
        local_score = R_2

        # Step 4: Global Scoring
        R_3 = torch.matmul(Adj, x4)
        global_score = self.lin2(R_3)

        # Step 5: Combine local and global scores
        ranking_scores = torch.add(local_score, global_score)
        return ranking_scores


class GNN(torch.nn.Module):
    def __init__(self, input_feature, output_feature):
        super(GNN, self).__init__()
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.a = nn.Parameter(torch.empty(size=(1, output_feature)))
        self.sigmod = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.w.data,gain=1.414)
        # nn.init.xavier_uniform_(self.a.data,gain=1.414)
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, x, adj):
        # adj=torch.Tensor(adj.numpy()+np.identity(adj.shape[0]))
        adj = torch.FloatTensor(normalize_adj(sp.csr_matrix(adj) + sp.eye(adj.shape[0])).todense())
        x = torch.mm(adj, x)
        x = torch.mm(x, self.w)
        # x=x.add(self.a)
        x = torch.relu(x)
        return x


class GAE(nn.Module):  # 编码器
    def __init__(self, n_total_features, n_latent, p_drop=0.):
        super(GAE, self).__init__()
        self.n_total_features = n_total_features
        self.conv1 = GNN(self.n_total_features, 256)

        self.conv2 = GNN(128, 24)

        self.conv3 = GNN(256, n_latent)
        self.conv4 = GNN(n_latent, self.n_total_features)
        self.fc1 = torch.nn.Linear(n_latent, 256)
        self.fc2 = torch.nn.Linear(256, 1)

    def forward(self, x, adj):  # 实践中一般采取多层的GCN来编码

        x = self.conv1(x, adj)
        # x = self.conv2(x, adj)
        x = self.conv3(x, adj)  # 经过三层GCN后得到节点的表示
        A = self.fc1(x)  # 直接算点积
        A = self.fc2(A)
        # A = torch.sigmoid(A)
        return x, A


class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=10,
                            kernel_size=(3, 3),
                            stride=1,
                            padding=1),

            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))

        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=10,
                            out_channels=20,
                            kernel_size=(3, 3),
                            stride=1,
                            padding=1),

            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))

        )
        self.fc = torch.nn.Linear(980, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


class GNN(torch.nn.Module):
    def __init__(self, input_feature, output_feature):
        super(GNN, self).__init__()
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.a = nn.Parameter(torch.empty(size=(1, output_feature)))
        self.sigmod = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.w.data,gain=1.414)
        # nn.init.xavier_uniform_(self.a.data,gain=1.414)
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, x, adj):
        adj = torch.Tensor(adj.numpy() + np.identity(adj.shape[0]))
        # adj=torch.FloatTensor(normalize_adj(sp.csr_matrix(adj)+ sp.eye(adj.shape[0])).todense())
        x = torch.mm(adj, x)
        x = torch.mm(x, self.w)
        x = x.add(self.a)
        x = torch.relu(x)
        return x


class CGNN(torch.nn.Module):
    def __init__(self):
        super(CGNN, self).__init__()
        self.layer1 = CNNnet()
        self.layer2 = GNN(48, 6)
        self.layer3 = GNN(6, 12)
        self.fc = torch.nn.Linear(12, 1)

    def forward(self, x, adj):
        # x=self.layer1(x)
        x = self.layer2(x, adj)
        x = self.layer3(x, adj)
        # x=self.layer4(x,adj)

        x = self.fc(x.view(x.size(0), -1))
        x = torch.relu(x)
        x = x.flatten()

        return x

class ListMLE(nn.Module):
    def __init__(self):
        super(ListMLE, self).__init__()
    def forward(self, outputs, labels):
        scores = torch.zeros_like(outputs)
        for t in range(scores.size()[0]):
            scores[t] = torch.logcumsumexp(outputs[t, labels[t]], dim=0)
        loss = torch.mean(scores - outputs)
        return loss
