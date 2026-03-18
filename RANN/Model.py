import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# ==========================================
# 4.2.2 & 4.2.3 结构感知注意力层与依赖关系交互
# ==========================================
class RoleQueryAttentionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4):
        # 使用 'add' 聚合方式，对应公式中的 \sum
        super(RoleQueryAttentionConv, self).__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.head_dim = out_channels // heads

        assert out_channels % heads == 0, "out_channels 必须能被 heads 整除"

        # Q, K, V 投影矩阵 (对应公式中的 W_Q, W_K, W_V)
        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)

        # 输出融合矩阵 W_O
        self.lin_out = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        # x 为节点初始的 struc2vec 角色特征
        # 生成查询(需求)、键(身份)、值(控制力)
        query = self.lin_q(x).view(-1, self.heads, self.head_dim)
        key = self.lin_k(x).view(-1, self.heads, self.head_dim)
        value = self.lin_v(x).view(-1, self.heads, self.head_dim)

        # 开始在边上进行消息传递 (中心节点 i 发出 query，邻居 j 提供 key 和 value)
        out = self.propagate(edge_index, query=query, key=key, value=value)

        # 拼接多头并做线性投影
        out = out.view(-1, self.out_channels)
        return self.lin_out(out)

    def message(self, query_i, key_j, value_j, index, ptr, size_i):
        # query_i: 中心节点的查询向量 (需求)
        # key_j: 邻居节点的键向量 (供给)

        # 计算供需契合度得分 e_{ij} = (Q_i * K_j^T) / sqrt(d_k)
        alpha = (query_i * key_j).sum(dim=-1) / (self.head_dim ** 0.5)

        # 邻域内的 Softmax 归一化 (注意：index 代表中心节点 i，确保在局部邻域内求和)
        alpha = softmax(alpha, index, ptr, size_i)

        # 将非对称依赖权重 alpha_ij 乘上邻居的价值向量 V_j
        return value_j * alpha.unsqueeze(-1)


class RoleInteractionBlock(nn.Module):
    """包含多头注意力、残差连接与前馈网络(FFN)的完整交互块"""

    def __init__(self, embed_dim, heads=4):
        super(RoleInteractionBlock, self).__init__()
        self.attention = RoleQueryAttentionConv(embed_dim, embed_dim, heads=heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 逐位置前馈神经网络 FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index):
        # 1. 多头注意力与残差连接 (防过平滑)
        attn_out = self.attention(x, edge_index)
        x = self.norm1(x + attn_out)

        # 2. FFN 与残差连接
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ==========================================
# 4.2.4 拓扑重要性评分预测 (网络主体 RANN)
# ==========================================
class RANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=4, num_layers=2):
        super(RANN, self).__init__()
        # 初始维度对齐层
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 拓扑依赖交互层 (可堆叠多层以捕获高阶依赖)
        self.interaction_layers = nn.ModuleList([
            RoleInteractionBlock(hidden_dim, heads) for _ in range(num_layers)
        ])

        # 直通式打分网络 MLP
        self.scoring_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 严格约束分数 S_i 在 (0, 1) 之间
        )

    def forward(self, x, edge_index):
        # x 形状: [N, D] (即 4.2.1 节得到的 struc2vec 矩阵)
        h = self.input_proj(x)

        # 深度交互获取高阶结构表征 Z_i
        for layer in self.interaction_layers:
            h = layer(h, edge_index)

        # Z_i -> S_i
        scores = self.scoring_mlp(h).squeeze(-1)  # 形状: [N]
        return scores


# ==========================================
# 4.2.5 连通性约束联合优化损失 (纯无监督升级版)
# ==========================================
class JointDismantlingLoss(nn.Module):
    def __init__(self, lambda_macro=1.0, lambda_micro=1.0, alpha_l1=0.01):
        """
        移除了 K-shell 相关的 margin，变为纯无监督的宏微观双尺度物理目标
        """
        super(JointDismantlingLoss, self).__init__()
        self.lambda_macro = lambda_macro
        self.lambda_micro = lambda_micro
        self.alpha_l1 = alpha_l1

    def forward(self, S, adj_matrix):
        """
        参数:
        - S: 模型输出的攻击分数, Tensor [N], 值域在 (0, 1) 之间
        - adj_matrix: 原始网络的稠密邻接矩阵 A, Tensor [N, N] (无向图，对角线为0)
        """
        N = S.size(0)

        # ---------------------------------------------------------
        # 1. 宏观：谱连通性瓦解损失 (Macro Spectral Loss)
        # ---------------------------------------------------------
        # 计算保留概率 p_i = 1 - S_i
        p = 1.0 - S

        # 构建软掩码矩阵 \tilde{A} = P * A * P
        # 利用广播机制高效实现对角矩阵相乘，避免构造大型对角阵
        A_tilde = p.view(N, 1) * adj_matrix * p.view(1, N)

        # 求解 \tilde{A} 的最大特征值 (因为是实对称矩阵，使用 eigvalsh)
        # eigvalsh 返回升序排列的特征值，取最后一个即为最大特征值 (\lambda_{max})
        eigvals = torch.linalg.eigvalsh(A_tilde)
        macro_loss = eigvals[-1]

        # ---------------------------------------------------------
        # 2. 微观：局部邻域粉碎损失 (Micro Shattering Loss)
        # ---------------------------------------------------------
        # 目标公式: L_micro = \sum_i \prod_{j \in N(i)} (1 / (1 + S_j))
        # 定义单个节点的存活期望 E_j = 1 / (1 + S_j)
        # 为了避免 for 循环和连乘带来的数值下溢，采用对数转换技巧：
        # \prod E_j = exp( \sum \log E_j )

        # 2.1 计算对数存活期望: log(E_j) = -log(1 + S_j)
        log_E = -torch.log(1.0 + S)  # 形状: [N]

        # 2.2 聚合一阶邻域: 使用邻接矩阵直接完成求和 \sum_{j \in N(i)}
        # adj_matrix 的第 i 行表示 i 的所有邻居，矩阵乘法自动累加了邻居的 log_E
        sum_log_E = torch.matmul(adj_matrix, log_E.unsqueeze(-1)).squeeze(-1)  # 形状: [N]

        # 2.3 还原为连乘概率，并对全网节点求和
        micro_loss = torch.exp(sum_log_E).mean()

        # ---------------------------------------------------------
        # 3. 稀疏性惩罚 (Sparsity Penalty / L1 Budget)
        # ---------------------------------------------------------
        # 限制攻击预算，迫使模型将极高的攻击分数精准集中在极少数真正的拓扑枢纽上
        # 因为 S 已经由 Sigmoid 约束在 (0, 1) 之间，直接求均值等价于 L1 正则
        l1_loss = S.mean()

        # ---------------------------------------------------------
        # 4. 联合损失 (Total Loss)
        # ---------------------------------------------------------
        total_loss = (self.lambda_macro * macro_loss +
                      self.lambda_micro * micro_loss +
                      self.alpha_l1 * l1_loss)

        # 返回总损失的同时，返回各项指标以便在日志中监控博弈过程
        return total_loss, macro_loss, micro_loss, l1_loss