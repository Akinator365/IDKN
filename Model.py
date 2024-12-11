from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul

# Embedding Dimension 1-32-16-8
# Attention Head      8-4-2
class IDKN(torch.nn.Module):
    def __init__(self):
        super(IDKN, self).__init__()

        self.conv1 = GATConv( 1, 4, heads=8, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv2 = GATConv( 32, 4, heads=4, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv3 = GATConv( 16, 4, heads=2, concat=True, negative_slope=0.2, dropout=0.2)

        self.lin1 = Linear(7, 1, bias=True)
        self.lin2 = Linear(8, 1, bias=False)

        self.activation = nn.ReLU()

    def forward(self, x1, x2, edge_index, num_nodes):
        fill_value = 1
        Adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1],
                                  sparse_sizes=(num_nodes, num_nodes))
        Adj_matrix = fill_diag(Adj_matrix, fill_value)

        Adj = F.normalize(Adj_matrix.to_dense(), p=1, dim=1)

        # Feature Scoring
        init_score = self.lin1(x1)
        init_score = self.activation(init_score)

        # Encoding Representation
        x3 = self.conv1(init_score, edge_index)
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
        #normalied_degree = x[:, 0].view(-1, 1)
        #local_score = R_2 + normalied_degree
        local_score = R_2

        # Global Socring
        R_3 = torch.matmul(Adj, x6)
        global_score = self.lin2(R_3)

        ranking_scores = torch.add(local_score, global_score)
        return ranking_scores



class IDKN_cat(torch.nn.Module):
    def __init__(self):
        super(IDKN_cat, self).__init__()

        # GATConv layers
        self.conv1 = GATConv(8, 4, heads=8, concat=True, negative_slope=0.2, dropout=0.1)  # 8 = new input dimension
        self.conv2 = GATConv(32, 4, heads=4, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv3 = GATConv(16, 4, heads=2, concat=True, negative_slope=0.2, dropout=0.2)

        # Linear layers
        self.lin1 = Linear(7 + 10, 8, bias=True)  # Input dimension updated to include role vector (e.g., 7 + 3)
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
        init_score = self.lin1(x_combined)

        # Encoding Representation
        x3 = self.conv1(init_score, edge_index)
        x4 = self.conv2(x3, edge_index)
        x5 = self.conv3(x4, edge_index)
        x6 = F.dropout(x5, p=0.3, training=self.training)

        # Local Scoring
        R = torch.matmul(x6, x6.t())
        R_1 = torch.mul(R, Adj)
        R_2 = torch.sum(R_1, dim=1, keepdim=True)
        local_score = R_2

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
        self.conv1 = GATConv(8, 4, heads=8, concat=True, negative_slope=0.2, dropout=0.1)  # 8 = cross-attention output dim
        self.conv2 = GATConv(32, 4, heads=4, concat=True, negative_slope=0.2, dropout=0.1)
        self.conv3 = GATConv(16, 4, heads=2, concat=True, negative_slope=0.2, dropout=0.2)

        # Attention parameters
        self.query_linear = Linear(8, 16)  # Transform x1 (中心性特征)
        self.key_linear = Linear(10, 16)   # Transform x2 (角色特征)
        self.value_linear = Linear(18, 16)

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
        fused_features = attention_weights * value

        # Step 2: GAT layers
        x1 = self.conv1(fused_features, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = F.dropout(x3, p=0.3, training=self.training)

        # Step 3: Local Scoring
        fill_value = 1
        Adj_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes))
        Adj_matrix = Adj_matrix.to_dense()
        Adj_matrix = F.normalize(Adj_matrix, p=1, dim=1)

        R = torch.matmul(x4, x4.t())
        R_1 = torch.mul(R, Adj_matrix)
        R_2 = torch.sum(R_1, dim=1, keepdim=True)
        local_score = R_2

        # Step 4: Global Scoring
        R_3 = torch.matmul(Adj_matrix, x4)
        global_score = self.lin2(R_3)

        # Step 5: Combine local and global scores
        ranking_scores = torch.add(local_score, global_score)
        return ranking_scores