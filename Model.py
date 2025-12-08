import numpy as np
import scipy as sp
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
import torch.nn as nn
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
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

# 传统GAE
class TraditionalGAE(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes

        # 编码器（保留原结构）
        self.conv1 = GCNConv(num_nodes, 512)
        self.conv2 = GCNConv(512, 128)  # 直接输出嵌入维度

        self.register_buffer('x', torch.eye(num_nodes))

    def forward(self, adj):
        x = self.x
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)

        # 编码器
        x = F.relu(self.conv1(x, edge_index))
        z = F.relu(self.conv2(x, edge_index))  # 输出嵌入 [N, embed_dim]
        return z  # 仅返回嵌入

# 适用于重构一维特征（如度值）的GAE
class RevisedGAE(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        # 编码器（遵循论文结构）
        self.conv1 = GCNConv(num_nodes, 512)  # 自动处理自环和归一化
        self.conv2 = GCNConv(512, 128)  # d/4=64

        # 解码器
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 1)

        self.register_buffer('x', torch.eye(num_nodes))  # 注册为缓冲区

    def forward(self, adj):
        x = self.x  # 直接使用缓存的单位矩阵
        """
        输入:
        x: 单位矩阵 (n x n)
        adj: 原始邻接矩阵 (n x n)

        返回:
        x: 节点嵌入 (n x d/4)
        A: 重建的节点度预测 (n x 1)
        """
        # 转换邻接矩阵为PyG需要的边索引格式
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)  # 确保自环存在

        # 编码（每层动态处理A~）
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # 解码
        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)

        # 修改点 2: 加上 ReLU，确保预测的度值非负
        return x, F.relu(A)


class GDNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, negative_slope=0.2, bias=True):
        # 聚合方式依然是 add，因为我们要累加接收到的扩散量
        super(GDNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope

        # 对应论文 Eq(2): LeakyReLU(W1*h_i + W2*h_j)
        # linear_src 对应 W1 (源节点变换), linear_dst 对应 W2 (目标节点变换)
        self.linear_src = Linear(in_channels, out_channels, bias=False)
        self.linear_dst = Linear(in_channels, out_channels, bias=False)

        # 对应论文 Eq(2) 中的向量 a (虽然论文没显式写出向量a的点积，
        # 但通常 LeakyReLU 后接一个向量点积来映射为标量 score，
        # 或者如论文描述直接用 LeakyReLU 输出作为 logits，这里参照标准 GAT 实现加一个 attn 向量)
        # 如果严格按照论文文字 "LeakyReLU(W1h + W2h)" 作为一个标量，
        # 意味着 out_channels 应该是 1。但通常为了表达能力，这里隐含了 attention vector。
        self.attn_vec = Parameter(torch.Tensor(1, out_channels))

        # 对应论文 Eq(4): W3 * m + b3
        self.lin_update = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_src.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.lin_update.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_vec, gain=gain)

    def forward(self, x, edge_index):
        # x: (N, in_channels)
        # edge_index: (2, E) -> [source_nodes, target_nodes]

        # 1. 保存源节点索引，用于后续的 Softmax 归一化
        # edge_index[0] 是源节点索引 (j -> i 中的 j)
        src_index = edge_index[0]

        # 2. 开始消息传递
        # 我们将 src_index 传递给 propagate，它会被透传给 message 函数
        out = self.propagate(edge_index, x=x, src_index=src_index)

        # 3. 对应论文 Eq(4): update 步骤
        # out 已经是聚合后的 m_i
        out = self.lin_update(out)

        # 4. 激活 (Activation - Eq 4)
        # 最后进行激活 sigma(.)
        return F.elu(out)

    def message(self, x_i, x_j, src_index, index, ptr, size_i):
        # x_j: 源节点特征 (E, out_channels)
        # x_i: 目标节点特征 (E, out_channels)
        # src_index: 源节点索引 (E, ) -> 用于 source-wise softmax
        # index: 目标节点索引 (E, ) -> PyG 默认传进来的，用于 scatter 聚合

        # --- 步骤 A: 计算 Attention Score (对应 Eq 2) ---

        # 变换特征
        h_src = self.linear_src(x_j)  # W1 * h_source
        h_dst = self.linear_dst(x_i)  # W2 * h_target

        # 计算系数: LeakyReLU(W1*h_src + W2*h_dst)
        # 形状: (E, out_channels)
        feat_sum = h_src + h_dst
        score = F.leaky_relu(feat_sum, self.negative_slope)

        # 将向量投影为标量: (E, out_channels) * (1, out_channels) -> sum -> (E, )
        score = (score * self.attn_vec).sum(dim=-1)

        # --- 步骤 B: 计算 Diffusion Weight (对应 Eq 3) ---

        # !!! 关键修改点 !!!
        # 标准 GAT 使用 softmax(score, index) -> 对指向同一目标节点的边归一化 (In-degree)
        # GDN 需要 softmax(score, src_index) -> 对来自同一源节点的边归一化 (Out-degree)

        alpha = softmax(score, src_index, ptr, None)  # 这里的 None 代表 num_nodes，通常可以省略

        # alpha 是扩散权重 w_{ij} (或 w_{ji})

        # --- 步骤 C: 消息加权 ---
        # 源节点特征 x_j 乘以它流向目标的比例 alpha
        return x_j * alpha.view(-1, 1)


# === 端到端预测模型 ===
class GDN_SIR_Predictor(nn.Module):
    def __init__(self, hidden_dim=64):
        super(GDN_SIR_Predictor, self).__init__()

        # 输入策略：
        # 策略 A: 既然是 SIR 模拟，假设每个节点初始状态一样，用 Embedding 模拟 "1单位能量"
        # 使用 learnable embedding 比纯 torch.ones 表达能力稍强，
        # 但如果为了严格模拟 DCRS 论文，可以改回固定全 1 输入。
        # 这里我使用一个单一的 learnable vector 广播到所有节点，模拟“初始均匀病毒”
        # [修复 1]: 引入度值编码器 (打破对称性的关键)
        self.degree_encoder = nn.Linear(1, hidden_dim)

        # [修复 2]: 使用随机初始化而不是全 1
        self.initial_val = nn.Parameter(torch.randn(1, hidden_dim))

        # GDN 层 (堆叠 4 层以捕获 4-hop 传播)
        self.gdn1 = GDNConv(hidden_dim, hidden_dim)
        self.gdn2 = GDNConv(hidden_dim, hidden_dim)
        self.gdn3 = GDNConv(hidden_dim, hidden_dim)
        self.gdn4 = GDNConv(hidden_dim, 1)

    def forward(self, data):
        # 获取输入
        edge_index = data.edge_index

        # 动态获取当前 batch 的总节点数
        # PyG 的 DataLoader 会把多个图拼成一个大图，data.num_nodes 是当前 batch 的总节点数
        if hasattr(data, 'num_nodes') and data.num_nodes is not None:
            curr_num_nodes = data.num_nodes
        else:
            # 如果没有 num_nodes 属性，尝试从 edge_index 推断
            curr_num_nodes = edge_index.max().item() + 1

        row, col = edge_index
        deg = degree(col, curr_num_nodes, dtype=torch.float).view(-1, 1)
        deg = deg.to(edge_index.device)
        # 对度值做 Log 平滑，防止数值过大
        deg = torch.log(deg + 1)

        # [步骤 2]: 编码度值
        deg_emb = F.elu(self.degree_encoder(deg))  # (N, hidden_dim)

        # 1. 构造初始特征 (N, D)
        # 将 (1, D) 的初始向量扩展为 (Batch_Total_Nodes, D)
        x = self.initial_val.expand(curr_num_nodes, -1) + deg_emb

        # 2. 添加自环 (Self-loops)
        # GDN 模拟扩散，必须有自环，否则能量会全部流失给邻居，自己不保留
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=curr_num_nodes)

        # 3. 逐层传播 (Layer-wise Propagation)
        x = F.elu(self.gdn1(x, edge_index_loop))  # Layer 1
        x = F.elu(self.gdn2(x, edge_index_loop))  # Layer 2
        x = F.elu(self.gdn3(x, edge_index_loop))  # Layer 3

        # 4. 最终评分
        # 最后一层输出 (N, 1)
        score = self.gdn4(x, edge_index_loop)

        # 5. 激活函数
        # SIR 影响力恒 > 0，使用 ReLU 保证非负
        return F.softplus(score).squeeze(-1)  # 输出 (N, )


# 适用于重构多维特征的（node2vec、struct2vec）GAE
class optimitzedGAE(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        # 编码器（遵循论文结构）
        self.conv1 = GCNConv(num_nodes, 512)  # 自动处理自环和归一化
        self.conv2 = GCNConv(512, 128)  # d/4=64
        # self.res_linear = nn.Linear(512, 128)  # 残差投影层
        # self.dropout = nn.Dropout(0.2)  # 防止过拟合

        # 解码器
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        # self.fc1 = nn.Linear(128, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)

        self.register_buffer('x', torch.eye(num_nodes))  # 注册为缓冲区

    def forward(self, adj):
        x = self.x  # 直接使用缓存的单位矩阵
        """
        输入:
        x: 单位矩阵 (n x n)
        adj: 原始邻接矩阵 (n x n)

        返回:
        x: 节点嵌入 (n x d/4)
        A: 重建的节点度预测 (n x 1)
        """
        # 转换邻接矩阵为PyG需要的边索引格式
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)  # 确保自环存在

        # 编码（每层动态处理A~）
        x = F.relu(self.conv1(x, edge_index))
        # x = self.dropout(x)
        # x_conv1 = x
        x = F.relu(self.conv2(x, edge_index))

        # 残差连接（通过投影调整维度）
        # x = x + self.res_linear(x_conv1)  # [N,128] + [N,128]

        # 解码
        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)
        # A = F.relu(A)
        # A = self.fc3(A)

        return x, F.normalize(A, p=2, dim=1)


# 适用于重构一维特征（如度值）的GAE,加入了残差模块
class resGAE(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        # 编码器（遵循论文结构）
        self.conv1 = GCNConv(num_nodes, 512)  # 自动处理自环和归一化
        self.conv2 = GCNConv(512, 128)  # d/4=64

        self.res_linear = nn.Linear(512, 128)  # 残差投影层
        self.dropout = nn.Dropout(0.2)  # 防止过拟合
        # 解码器
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)

        self.register_buffer('x', torch.eye(num_nodes))  # 注册为缓冲区

    def forward(self, adj):
        x = self.x  # 直接使用缓存的单位矩阵
        """
        输入:
        x: 单位矩阵 (n x n)
        adj: 原始邻接矩阵 (n x n)

        返回:
        x: 节点嵌入 (n x d/4)
        A: 重建的节点度预测 (n x 1)
        """
        # 转换邻接矩阵为PyG需要的边索引格式
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)  # 确保自环存在

        # 编码（每层动态处理A~）
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x_conv1 = x
        x = F.relu(self.conv2(x, edge_index))

        # 残差连接（通过投影调整维度）
        x = x + self.res_linear(x_conv1)  # [N,128] + [N,128]

        # 解码
        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)

        return x, A


class res_rand_GAE(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes

        # ===== 可学习随机初始化嵌入向量 =====
        self.node_emb = nn.Parameter(torch.empty(num_nodes, 128))  # 用 nn.Parameter 替代 one-hot
        nn.init.xavier_uniform_(self.node_emb)  # 可选：xavier_normal_ 也可以

        # 编码器（遵循论文结构）
        self.conv1 = GCNConv(128, 512)  # 输入维度仍为 num_nodes（如果你希望改小可以一起调整）
        self.conv2 = GCNConv(512, 128)

        self.res_linear = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.2)

        # 解码器
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, adj):
        x = self.node_emb  # 使用可学习嵌入

        # 转换邻接矩阵为PyG需要的边索引格式
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)

        # 编码
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x_conv1 = x
        x = F.relu(self.conv2(x, edge_index))

        # 残差连接
        x = x + self.res_linear(x_conv1)

        # 解码
        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)

        return x, A


class rand_GAE(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes

        # ===== 可学习随机初始化嵌入向量 =====
        self.node_emb = nn.Parameter(torch.empty(num_nodes, 128))  # 用 nn.Parameter 替代 one-hot
        nn.init.xavier_uniform_(self.node_emb)  # 可选：xavier_normal_ 也可以

        # 编码器（遵循论文结构）
        self.conv1 = GCNConv(128, 512)  # 输入维度仍为 num_nodes（如果你希望改小可以一起调整）
        self.conv2 = GCNConv(512, 128)

        # 解码器
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, adj):
        x = self.node_emb  # 使用可学习嵌入

        # 转换邻接矩阵为PyG需要的边索引格式
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)

        # 编码
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # 解码
        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)

        return x, A


# 适用于重构多维特征的（node2vec、struct2vec）GAE，加入了残差模块
class resvecGAE(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        # 编码器（遵循论文结构）
        self.conv1 = GCNConv(num_nodes, 512)  # 自动处理自环和归一化
        self.conv2 = GCNConv(512, 128)  # d/4=64
        self.res_linear = nn.Linear(512, 128)  # 残差投影层
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

        # 解码器
        #self.fc1 = nn.Linear(128, 256)
        #self.fc2 = nn.Linear(256, 128)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)

        self.register_buffer('x', torch.eye(num_nodes))  # 注册为缓冲区

    def forward(self, adj):
        x = self.x  # 直接使用缓存的单位矩阵
        """
        输入:
        x: 单位矩阵 (n x n)
        adj: 原始邻接矩阵 (n x n)

        返回:
        x: 节点嵌入 (n x d/4)
        A: 重建的节点度预测 (n x 1)
        """
        # 转换邻接矩阵为PyG需要的边索引格式
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)  # 确保自环存在

        # 编码（每层动态处理A~）
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x_conv1 = x
        x = F.relu(self.conv2(x, edge_index))

        # 残差连接（通过投影调整维度）
        x = x + self.res_linear(x_conv1)  # [N,128] + [N,128]

        # 解码
        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)

        return x, F.normalize(A, p=2, dim=1)

# 适用于重构一维特征（如度值）的GAE
class struct_start_GAE(nn.Module):
    def __init__(self, num_nodes, struct2vec_emb):
        super().__init__()
        self.num_nodes = num_nodes
        # ==== 核心修改：固定 struct2vec 嵌入 ====
        # 使用 register_buffer 而非 Parameter
        self.register_buffer('struct_emb', struct2vec_emb.float())  # [N, 128]
        # 编码器（遵循论文结构）
        self.conv1 = GCNConv(128, 512)  # 自动处理自环和归一化
        self.conv2 = GCNConv(512, 128)  # d/4=64

        # 解码器
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, adj):
        x = self.struct_emb  # 直接使用缓存的单位矩阵
        """
        输入:
        x: 单位矩阵 (n x n)
        adj: 原始邻接矩阵 (n x n)

        返回:
        x: 节点嵌入 (n x d/4)
        A: 重建的节点度预测 (n x 1)
        """
        # 转换邻接矩阵为PyG需要的边索引格式
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)  # 确保自环存在

        # 编码（每层动态处理A~）
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # 解码
        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)
        return x, A


# 适用于重构多维特征的（node2vec、struct2vec）GAE
class learnableGAE(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        # ==== 可学习的输入特征 + Xavier 初始化 ====
        self.node_emb = nn.Parameter(torch.empty(num_nodes, 256))  # 先定义空张量
        nn.init.xavier_normal_(self.node_emb)  # 应用 Xavier 初始化

        # 编码器（遵循论文结构）
        self.conv1 = GCNConv(256, 512)  # 自动处理自环和归一化
        self.conv2 = GCNConv(512, 128)  # d/4=64
        self.res_linear = nn.Linear(512, 128)  # 残差投影层
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

        # 解码器
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, adj):
        """
        输入:
        adj: 原始邻接矩阵 (n x n)

        返回:
        x: 节点嵌入 (n x d/4)
        A: 重建的节点度预测 (n x 1)
        """
        # ==== 可学习的输入特征 ====
        x = self.node_emb  # [N, input_dim]

        # 转换邻接矩阵为PyG需要的边索引格式
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)  # 确保自环存在

        # 编码（每层动态处理A~）
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x_conv1 = x
        x = F.relu(self.conv2(x, edge_index))

        # 残差连接（通过投影调整维度）
        x = x + self.res_linear(x_conv1)  # [N,128] + [N,128]

        # 解码
        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)

        return x, F.normalize(A, p=2, dim=1)


class structembGAE(nn.Module):
    def __init__(self, num_nodes, struct2vec_emb):
        super().__init__()
        self.num_nodes = num_nodes

        # ==== 核心修改 1：用 struct2vec 嵌入替换单位矩阵 ====
        # 注册 struct2vec 嵌入为可学习参数
        self.node_emb = nn.Parameter(struct2vec_emb)

        # ==== 核心修改 2：调整编码器输入维度 ====
        # 原 GCNConv(num_nodes, 512) → 改为 struct2vec 嵌入维度
        input_dim = struct2vec_emb.shape[1]  # 自动获取维度（假设 struct2vec_emb 是 [N, D]）
        self.conv1 = GCNConv(input_dim, 512)

        # 其他层保持不变
        self.conv2 = GCNConv(512, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, adj):
        # ==== 核心修改 3：使用 struct2vec 嵌入作为输入 ====
        x = self.node_emb  # [num_nodes, struct2vec_dim]

        # 后续处理保持不变
        edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        A = self.fc1(x)
        A = F.relu(A)
        A = self.fc2(A)
        return x, A


class EnhancedGAE(nn.Module):
    def __init__(self, input_dim, latent_dim=48):
        super().__init__()
        # 编码器
        self.conv1 = GCNConv(input_dim, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, latent_dim)

        # 解码器
        self.degree_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 正则化组件
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.skip_conn = nn.Linear(input_dim, latent_dim)

    def forward(self, x, edge_index):
        identity = self.skip_conn(x)

        # 编码过程
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        z = self.conv3(x, edge_index) + identity

        # 解码过程
        adj_recon = torch.sigmoid(torch.mm(z, z.t()))
        degree_pred = self.degree_predictor(z)

        return z, adj_recon, degree_pred.squeeze()


class CGNN_more_fc(torch.nn.Module):
    def __init__(self):
        super(CGNN_more_fc, self).__init__()
        # CNN层
        self.layer1 = GCNConv(128, 512)  # 使用GCNConv替代原始GNN层
        #self.layer2 = GCNConv(256, 128)  # 使用GCNConv替代原始GNN层
        self.layer3 = GCNConv(512, 128)  # 输入/输出特征维度需匹配
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 1)

        # 更精细的初始化
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out')


    def forward(self, x, edge_index):
        # 1. 使用edge_index进行图卷积
        x = self.layer1(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        #x = self.layer2(x, edge_index)
        #x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.layer3(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)


        # 2. 修改全连接层处理方式
        x = self.fc1(x)  # [num_nodes, 1]
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.fc2(x)
        x = x.squeeze(-1)  # [num_nodes]
        return x

class CGNN_New(torch.nn.Module):
    def __init__(self):
        super(CGNN_New, self).__init__()
        # CNN层
        self.layer1 = GCNConv(128, 256)  # 使用GCNConv替代原始GNN层
        self.layer2 = GCNConv(256, 128)  # 使用GCNConv替代原始GNN层
        self.layer3 = GCNConv(128, 32)  # 输入/输出特征维度需匹配
        self.fc = torch.nn.Linear(32, 1)

        # 更精细的初始化
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')

    def forward(self, x, edge_index):
        # 1. 使用edge_index进行图卷积
        x = self.layer1(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.layer2(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.layer3(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)

        # 2. 修改全连接层处理方式
        x = self.fc(x)  # [num_nodes, 1]
        x = x.squeeze(-1)  # [num_nodes]
        return x

class CGNN_GAT(torch.nn.Module):
    def __init__(self):
        super(CGNN_GAT, self).__init__()
        # CNN层
        self.layer1 = GCNConv(128, 512)
        self.layer3 = GATConv(512, 64, heads=4, concat=False)  # 输入/输出特征维度需匹配
        self.fc = torch.nn.Linear(64, 1)

        # 更精细的初始化
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')

    def forward(self, x, edge_index):
        # 1. 使用edge_index进行图卷积
        x = self.layer1(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.layer3(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)

        # 2. 修改全连接层处理方式
        x = self.fc(x)  # [num_nodes, 1]
        x = x.squeeze(-1)  # [num_nodes]
        return x
