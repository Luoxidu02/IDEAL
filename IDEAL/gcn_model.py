
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
#from main import args  # 或 from train import args
from torch_geometric.nn import GCNConv
from torch.nn import MultiheadAttention

# 文件: gcn_model.py
# 在文件末尾添加以下新类

import torch
from torch.autograd import Function
import torch.nn as nn


# 文件: gcn_model.py
# 在 GCNWithDecoderWrapper_cam 类的定义之后，添加以下新类
# gcn_model.py
import torch
# ...
from torch_geometric.nn import GCNConv, GATConv # <--- 确认已导入

# ... 已定义的 GCN_pyg 类 ...

# 【确保这个 GAT 模型类存在】
# gcn_model.py

from torch.nn import LayerNorm  # 【【【在文件顶部导入 LayerNorm】】】


class GAT_pyg(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2, heads=8):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)

        # 【【【添加两个 LayerNorm 层】】】
        self.norm1 = LayerNorm(hidden_dim * heads)
        self.norm2 = LayerNorm(hidden_dim)

        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)  # 【【【在激活函数前使用 LayerNorm】】】
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)  # 【【【在最终激活函数前使用 LayerNorm】】】
        x = F.elu(x)

        return x


class GCNWithDecoderWrapperBatched(torch.nn.Module):
    """
    一个为 GNNExplainer 设计的、支持内部批处理的 Wrapper。
    它解决了在大型计算子图上一次性解码所有边导致的显存爆炸问题。
    """

    def __init__(self, gcn_model, decoder, microbe_offset, batch_size=4096):
        super().__init__()
        self.gcn = gcn_model
        self.decoder = decoder
        self.microbe_offset = microbe_offset
        self.batch_size = batch_size  # 内部处理边的批量大小

    def forward(self, x, edge_index, edge_weight=None):
        # 1. 使用 GCN 模型获取子图中所有节点的嵌入
        # 这一步的计算量是可控的，只涉及子图的节点
        embeddings, _ = self.gcn(x, edge_index, edge_weight=edge_weight)

        # 2. 对边进行批处理解码，避免一次性创建巨大张量
        num_edges = edge_index.size(1)

        # 如果边的数量不多，或者只有一条边（兼容BatchNorm1d），就直接处理
        if num_edges <= self.batch_size:
            src_emb = embeddings[edge_index[0]]
            dst_emb = embeddings[edge_index[1]]
            # 兼容单条边输入时 BatchNorm1d 的情况
            if src_emb.dim() == 1:
                src_emb = src_emb.unsqueeze(0)
            if dst_emb.dim() == 1:
                dst_emb = dst_emb.unsqueeze(0)
            return self.decoder(src_emb, dst_emb)

        # 当边数量非常多时，启动批处理
        outputs = []
        for i in range(0, num_edges, self.batch_size):
            # a. 获取当前批次的边索引
            batch_edge_index = edge_index[:, i: i + self.batch_size]

            # b. 提取当前批次边的嵌入
            src_emb = embeddings[batch_edge_index[0]]
            dst_emb = embeddings[batch_edge_index[1]]

            # c. 对当前批次进行解码
            batch_out = self.decoder(src_emb, dst_emb)
            outputs.append(batch_out)

        # 3. 将所有批次的结果拼接成一个完整的输出张量
        return torch.cat(outputs, dim=0)


#动态可组合注意力
class DynamicCombinationAttention(nn.Module):
    def __init__(self, num_inputs, input_dim):
        super().__init__()
        # 每个输入特征有一个可学习的标量权重
        self.raw_weights = nn.Parameter(torch.ones(num_inputs))
        self.input_dim = input_dim

    def forward(self, x):  # x: [batch, num_inputs, dim]
        # softmax归一化权重，shape: [num_inputs]
        weights = F.softmax(self.raw_weights, dim=0)
        # [1, num_inputs, 1]
        weights = weights.view(1, -1, 1)
        # [batch, num_inputs, dim] * [1, num_inputs, 1] => [batch, num_inputs, dim]
        weighted_x = x * weights
        # 融合
        out = weighted_x.sum(dim=1)  # [batch, dim]
        return out, weights

class GatedAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Linear(input_dim, 1)

    def forward(self, x):  # x: [batch, seq, dim]
        # 得到门控分数
        scores = torch.sigmoid(self.gate(x))  # [batch, seq, 1]
        # 权重归一化
        weights = torch.softmax(scores, dim=1)  # [batch, seq, 1]
        # 加权求和
        out = (x * weights).sum(dim=1)  # [batch, dim]
        return out, weights  # 返回门控权重可用于分析


class FeatureMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super(FeatureMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def normalize_adj(adj):
    """
    numpy实现的对称归一化
    """
    adj = adj + np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

# def normalize_adj_torch(adj):
#     """
#     torch实现的对称归一化
#     """
#     adj = adj + torch.eye(adj.size(0), device=adj.device)
#     degree = torch.sum(adj, dim=1)
#     d_inv_sqrt = torch.pow(degree, -0.5)
#     d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
#     D_inv_sqrt = torch.diag(d_inv_sqrt)
#     return D_inv_sqrt @ adj @ D_inv_sqrt

class MLPDecoder(nn.Module):
    # def __init__(self, embed_dim, hidden_dim=64):
    #     super(MLPDecoder, self).__init__()
    #     self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
    #     self.fc2 = nn.Linear(hidden_dim, 1)
    #
    # def forward(self, drug_emb, microbe_emb):
    #     # 拼接药物和微生物的嵌入
    #     x = torch.cat([drug_emb, microbe_emb], dim=1)
    #     x = torch.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x.squeeze(1)  # 输出logits（一维）






    # def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.2):
    #     super(MLPDecoder, self).__init__()
    #     self.mlp = nn.Sequential(
    #         nn.Linear(input_dim * 2, hidden_dim1),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_rate),
    #         nn.Linear(hidden_dim1, hidden_dim2),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_rate),
    #         nn.Linear(hidden_dim2, 1)
    #     )



    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, dropout_rate=0.4):
        super(MLPDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim3, 1)
        )


    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, drug_embeds, microbe_embeds):
        # 将药物和微生物的嵌入向量拼接在一起
        combined_embeds = torch.cat([drug_embeds, microbe_embeds], dim=1)

        # 通过MLP得到预测分数，并用squeeze(-1)将输出维度从 [batch_size, 1] 变为 [batch_size]
        return self.mlp(combined_embeds).squeeze(-1)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


from torch_geometric.nn import GCNConv

class GCN_pyg(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x
# gcn_model.py

# ==============================================================================
# =====================  全新的、更先进的 GCN 结构  ===========================
# ==============================================================================
from torch_geometric.nn import GCNConv, JumpingKnowledge
class GCN_Advanced(nn.Module):
    """
    一个集成了残差连接和跳跃连接的先进GCN模型。
    - num_layers: GCN层的数量。
    - hidden_dim: GCN层的隐藏维度。
    - in_dim: 输入特征的维度。
    - dropout: Dropout比率。
    - jk_mode: 跳跃连接的模式 ('cat', 'max', 'lstm')。'cat'通常效果最好。
    """

    def __init__(self, in_dim, hidden_dim, dropout, num_layers=3, jk_mode='cat'):
        super().__init__()
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # 添加批量归一化层，增强稳定性

        # 输入层
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 隐藏层
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 跳跃连接层
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)

        # 如果是'cat'模式，最终输出维度会变大，需要一个线性层映射回hidden_dim
        if jk_mode == 'cat':
            self.final_lin = nn.Linear(num_layers * hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        xs = []  # 存储每一层的输出，用于跳跃连接
        h = x

        for i in range(self.num_layers):
            h_prev = h
            h = self.convs[i](h, edge_index, edge_weight=edge_weight)
            h = self.bns[i](h)
            h = F.relu(h)

            # 残差连接 (仅当维度相同时)
            if h.shape == h_prev.shape:
                h = h + h_prev

            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            xs.append(h)

        # 应用跳跃连接
        h = self.jk(xs)

        # 'cat'模式后通过线性层
        if self.jk.mode == 'cat':
            h = self.final_lin(h)

        return h


#  (确保你已经删除了旧的 GCN_pyg 类)


#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nhid)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         # ----------------- 第一层GCN计算 -----------------
#         # 公式: H_1 = relu( D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² H_0 W_0 )
#         # 对应代码:
#         # x  -> H_0 (初始特征 features)
#         # adj -> D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² (归一化邻接矩阵)
#         # self.gc1 -> 包含 W_0 (第一层权重) 的层
#         # F.relu -> σ
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         # 此时的 x 就是 H_1 (第一层输出的节点嵌入)
#
#         # ----------------- 第二层GCN计算 -----------------
#         # 公式: H_2 = D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² H_1 W_1  (你的第二层没有激活函数)
#         # 对应代码:
#         # x  -> H_1 (第一层输出)
#         # adj -> D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² (还是同一个归一化邻接矩阵)
#         # self.gc2 -> 包含 W_1 (第二层权重) 的层
#         x = self.gc2(x, adj)
#         # 此时的 x 就是 H_2 (最终的节点嵌入)
#         return x  # 返回嵌入
# # gcn_model.py
# # gcn_model.py

def ensure_tensor_on_device(x, device):
    if not torch.is_tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device)
    else:
        return x.clone().detach().to(dtype=torch.float32, device=device)

class GCNWithMLP(nn.Module):
#     def __init__(self, drug_in_dim, drug_out_dim, microbe_dim, microbe_out_dim, gcn_hidden, dropout, mlp_hidden=2048,
#                  use_microbe_mlp=True   ,dataset_name="MDAD",f=2,scaling_factor=1.6,gnn_choice = 0 ):
#         super(GCNWithMLP, self).__init__()
#         gat_heads = 4  # GAT的多头注意力头数
#         if dataset_name == "DrugVirus":
#             self.DrugVirus_drug_mlp = FeatureMLP(175, 175, 256)  # 降维到175
#             self.DrugVirus_drug_mlp_1 = FeatureMLP(2048, 175, 1024)  # 保持维度
#             self.DrugVirus_microbe_mlp = FeatureMLP(95, 95, 256)  # 保持微生物维度
#
#         self.scaling_factor = scaling_factor  # 保存缩放因子
#
#         self.use_microbe_mlp = use_microbe_mlp  # 保存传入的参数
#         self.mlp = FeatureMLP(drug_in_dim, drug_out_dim, mlp_hidden)
#         self.aBiofilm_drug_mlp=FeatureMLP(1720, 1720, mlp_hidden)
#         self.aBiofilm_drug_mlp_1 = FeatureMLP(2048, 1720, mlp_hidden)
#         self.aBiofilm_microbe_mlp = FeatureMLP(140, 140, mlp_hidden)
#
#         self.MDAD_drug_mlp=FeatureMLP(1373, 1372, mlp_hidden)
#         self.MDAD_drug_mlp_1=FeatureMLP(2048, 1372, mlp_hidden)
#         self.MDAD_microbe_mlp=FeatureMLP(173, 170, mlp_hidden)
#         self.dataset_name=dataset_name
#         # 仅当 use_microbe_mlp 为 True 时才创建 microbe_mlp
#         if self.use_microbe_mlp:
#             self.microbe_mlp = FeatureMLP(microbe_dim, microbe_out_dim, mlp_hidden)
#         else:
#             self.microbe_mlp = None  # 不需要 microbe_mlp
#         self.f = f
#
#         # 创建 GCN 层，注意输出维度已根据是否使用 microbe_mlp 调整
#         self.gcn = GCN_pyg(drug_out_dim + (microbe_out_dim if self.use_microbe_mlp else microbe_dim), gcn_hidden, dropout)
#
# #################针对药物的注意力（开始）#################
#         # ===== 新增：三个可学习系数 =====
#         # self.alpha1 = nn.Parameter(torch.tensor(1.0))
#         # self.alpha2 = nn.Parameter(torch.tensor(1.0))
#         # self.alpha3 = nn.Parameter(torch.tensor(1.0))
#         self.alpha1 = nn.Parameter(torch.tensor(0.8))#药物的
#         self.alpha2 = nn.Parameter(torch.tensor(0.6))
#         self.alpha3 = nn.Parameter(torch.tensor(0.7))
#
#         # ===== 新增：transformer多头注意力融合 =====
#         # 假设每个特征维度一致
#         fusion_dim = drug_out_dim  # 假设降维后一致
#         if dataset_name == "DrugVirus":
#             nhead = 5
#             #
#             # fusion_dim = 175  # 新增：为DrugVirus设置正确的融合维度
#             # self.gcn = GCN_pyg(175 + 95, gcn_hidden, dropout)  # 修正GCN输入维度
#             #
#
#
#         if dataset_name == "MDAD":
#             nhead = 4
#             fusion_dim = 1372
#             self.gcn = GCN_pyg(1372 + 170, gcn_hidden,
#                                dropout)
#         if dataset_name == "aBiofilm":
#             nhead = 4
#             fusion_dim = 1720
#             self.gcn = GCN_pyg(1720 + 140, gcn_hidden,
#                            dropout)
#         self.fusion_transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=nhead, batch_first=True),
#             num_layers=1#注意力头是4（药物）
#         )
#         # ===== 修改：定义动态层注意力的Transformer（药物）=====
#         self.num_dynamic_layers = 3  # 定义动态层的数量，例如3层
#         self.fusion_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=nhead, batch_first=True, dropout=0.2, activation='gelu')
#             for _ in range(self.num_dynamic_layers)
#         ])
#         # 为每一层定义一个可学习的权重
#         self.layer_weights = nn.Parameter(torch.ones(self.num_dynamic_layers))
#         # == 新增: MHA 注意力 ==
#         self.mha_drug = MultiheadAttention(embed_dim=fusion_dim, num_heads=nhead, batch_first=True, dropout=0.1)
#         # 药物门控注意力
#         self.gated_attn_drug = GatedAttention(fusion_dim)
#         #药物动态可组合注意力
#         self.dynamic_comb_attn_drug = DynamicCombinationAttention(num_inputs=3, input_dim=fusion_dim)
#          #################针对药物的注意力（结束）#################
#         if self.f == 4:
#             # 假设药物三种特征维度分别如下（要和你的数据实际一致！）
#             drug_input_dims = [2048, 175, 175]  # drug_fg, drug_features, drug_bert
#             microbe_input_dims = [95, 95, 95]  # microbe_features, microbe_bert, microbe_path
#             drug_embed_dim = 128
#             microbe_embed_dim = 64
#             drug_output_dim = drug_out_dim  # 你原来降到多少就填多少
#             microbe_output_dim = microbe_out_dim
#
#             self.drug_token_fusion = FeatureTokenFusion(
#                 input_dims=drug_input_dims, embed_dim=drug_embed_dim, n_heads=4, n_layers=1, output_dim=drug_output_dim
#             )
#             self.microbe_token_fusion = FeatureTokenFusion(
#                 input_dims=microbe_input_dims, embed_dim=microbe_embed_dim, n_heads=4, n_layers=1,
#                 output_dim=microbe_output_dim
#             )
#
#         #################针对微生物的注意力（开始）#################
#         # ===== 新增：三个可学习系数 =====
#         self.alpha1_m = nn.Parameter(torch.tensor(0.8))
#         self.alpha2_m = nn.Parameter(torch.tensor(0.6))
#         self.alpha3_m = nn.Parameter(torch.tensor(0.7))
#         # ===== 新增：transformer多头注意力融合 =====
#         # 假设每个特征维度一致
#         if dataset_name == "DrugVirus":
#             nhead_m = 5
#             fusion_dim_m=95#药物的维度
#         if dataset_name == "MDAD":
#             nhead_m = 5
#             fusion_dim_m=170#是通过173降维得到的
#             # self.gcn = GCN_pyg(1373 + 170, gcn_hidden,
#             #                dropout)
#         if dataset_name == "aBiofilm":
#             nhead_m = 5
#             fusion_dim_m = 140
#         self.fusion_transformer_m = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=fusion_dim_m, nhead=nhead_m, batch_first=True),
#             num_layers=1
#         )
#         # ===== 修改：定义动态层注意力的Transformer（微生物）=====
#         self.num_dynamic_layers_m = 3 # 定义动态层的数量，例如3层
#         self.fusion_layers_m = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=fusion_dim_m, nhead=nhead_m, batch_first=True, dropout=0.1)
#             for _ in range(self.num_dynamic_layers_m)
#         ])
#         # 为每一层定义一个可学习的权重
#         self.layer_weights_m = nn.Parameter(torch.ones(self.num_dynamic_layers_m))
#         # == 新增: MHA 注意力 ==
#         self.mha_microbe = MultiheadAttention(embed_dim=fusion_dim_m, num_heads=nhead_m, batch_first=True, dropout=0.2)
#         # 微生物门控注意力
#         self.gated_attn_microbe = GatedAttention(fusion_dim_m)
#         #微生物动态可组合注意力
#         self.dynamic_comb_attn_microbe = DynamicCombinationAttention(num_inputs=3, input_dim=fusion_dim_m)
# #################针对药物的注意力（结束）#################
    # 这是 GCNWithMLP 类的 __init__ 方法
    def __init__(self, drug_in_dim, drug_out_dim, microbe_dim, microbe_out_dim, gcn_hidden, dropout, mlp_hidden=2048,
                 use_microbe_mlp=True, dataset_name="MDAD", f=2, scaling_factor=1.6):  # 注意：我移除了gnn_choice参数
        super(GCNWithMLP, self).__init__()

        # ==================== 【在这里设置开关】 ====================
        # 0: 使用 GCN 模型 (GCN_pyg)
        # 1: 使用 GAT 模型 (GAT_pyg)
        #2: 使用MLP模型(纯特征，无图结构)
        gnn_choice = 0  # <--- 在这里修改 0 或 1 即可切换模型！
        self.gnn_choice = gnn_choice  # <--- 【【【在这里添加这一行】】】
        # =========================================================

        # ==================== 【GAT专属超参数】 =====================
        # 如果选择GAT (gnn_choice = 1)，下面的参数会生效
        gat_heads = 2  # GAT的多头注意力头数
        # =========================================================

        # --- 从这里开始，是你原来的代码，我只修改了 self.gcn 的创建部分 ---
        if dataset_name == "DrugVirus":
            self.DrugVirus_drug_mlp = FeatureMLP(175, 175, 256)
            self.DrugVirus_drug_mlp_1 = FeatureMLP(2048, 175, 1024)
            self.DrugVirus_microbe_mlp = FeatureMLP(95, 95, 256)

        self.scaling_factor = scaling_factor
        self.use_microbe_mlp = use_microbe_mlp
        self.mlp = FeatureMLP(drug_in_dim, drug_out_dim, mlp_hidden)
        self.aBiofilm_drug_mlp = FeatureMLP(1720, 1720, mlp_hidden)
        self.aBiofilm_drug_mlp_1 = FeatureMLP(2048, 1720, mlp_hidden)
        self.aBiofilm_microbe_mlp = FeatureMLP(140, 140, mlp_hidden)
        self.MDAD_drug_mlp = FeatureMLP(1373, 1372, mlp_hidden)
        self.MDAD_drug_mlp_1 = FeatureMLP(2048, 1372, mlp_hidden)
        self.MDAD_microbe_mlp = FeatureMLP(173, 170, mlp_hidden)
        self.dataset_name = dataset_name
        if self.use_microbe_mlp:
            self.microbe_mlp = FeatureMLP(microbe_dim, microbe_out_dim, mlp_hidden)
        else:
            self.microbe_mlp = None
        self.f = f

        # 【【【 核心修改部分：替换原来的 self.gcn 创建逻辑 】】】
        # 1. 先根据数据集确定 GNN 的输入维度和注意力头的数量
        if dataset_name == "DrugVirus":
            nhead = 5
            fusion_dim = 175
            gnn_input_dim = 175 + 95
        elif dataset_name == "MDAD":
            nhead = 4
            fusion_dim = 1372
            gnn_input_dim = 1372 + 170
        elif dataset_name == "aBiofilm":
            nhead = 4
            fusion_dim = 1720
            gnn_input_dim = 1720 + 140
        else:  # 默认情况
            nhead = 4
            fusion_dim = drug_out_dim
            gnn_input_dim = drug_out_dim + (microbe_out_dim if self.use_microbe_mlp else microbe_dim)

        # 2. 根据 gnn_choice 开关来实例化 self.gcn
        if gnn_choice == 0:
            print(f"--- Model Switch: Using GCN (GCN_pyg) for dataset {dataset_name} ---")
            self.gcn = GCN_pyg(gnn_input_dim, gcn_hidden, dropout)
        elif gnn_choice == 1:
            print(f"--- Model Switch: Using GAT with {gat_heads} heads for dataset {dataset_name} ---")
            self.gcn = GAT_pyg(gnn_input_dim, gcn_hidden, dropout=dropout, heads=gat_heads)
            # 【【【在这里添加 gnn_choice == 2 的分支】】】
        elif gnn_choice == 2:
            print(f"--- Model Switch: Using MLP Encoder (no graph structure) for dataset {dataset_name} ---")
            # MLP模型是一个简单的全连接网络，它接收拼接后的节点特征X，输出嵌入
            # 输入维度是gnn_input_dim，输出维度是gcn_hidden，以和其他模型保持一致
            self.gcn = nn.Sequential(
                nn.Linear(gnn_input_dim, gcn_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gcn_hidden, gcn_hidden)
            )
        else:
            raise ValueError("Invalid gnn_choice. Must be 0 (GCN), 1 (GAT), or 2 (MLP).")
        # 【【【 核心修改结束 】】】

        # --- 后面的代码保持不变 ---
        self.alpha1 = nn.Parameter(torch.tensor(0.8))
        self.alpha2 = nn.Parameter(torch.tensor(0.6))
        self.alpha3 = nn.Parameter(torch.tensor(0.7))

        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=nhead, batch_first=True),
            num_layers=1
        )
        self.num_dynamic_layers = 3
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=nhead, batch_first=True, dropout=0.2,
                                       activation='gelu')
            for _ in range(self.num_dynamic_layers)
        ])
        self.layer_weights = nn.Parameter(torch.ones(self.num_dynamic_layers))
        self.mha_drug = MultiheadAttention(embed_dim=fusion_dim, num_heads=nhead, batch_first=True, dropout=0.1)
        self.gated_attn_drug = GatedAttention(fusion_dim)
        self.dynamic_comb_attn_drug = DynamicCombinationAttention(num_inputs=3, input_dim=fusion_dim)

        if self.f == 4:
            drug_input_dims = [2048, 175, 175]
            microbe_input_dims = [95, 95, 95]
            drug_embed_dim = 128
            microbe_embed_dim = 64
            drug_output_dim = drug_out_dim
            microbe_output_dim = microbe_out_dim

            self.drug_token_fusion = FeatureTokenFusion(
                input_dims=drug_input_dims, embed_dim=drug_embed_dim, n_heads=4, n_layers=1, output_dim=drug_output_dim
            )
            self.microbe_token_fusion = FeatureTokenFusion(
                input_dims=microbe_input_dims, embed_dim=microbe_embed_dim, n_heads=4, n_layers=1,
                output_dim=microbe_output_dim
            )

        self.alpha1_m = nn.Parameter(torch.tensor(0.8))
        self.alpha2_m = nn.Parameter(torch.tensor(0.6))
        self.alpha3_m = nn.Parameter(torch.tensor(0.7))

        if dataset_name == "DrugVirus":
            nhead_m = 5
            fusion_dim_m = 95
        if dataset_name == "MDAD":
            nhead_m = 5
            fusion_dim_m = 170
        if dataset_name == "aBiofilm":
            nhead_m = 5
            fusion_dim_m = 140
        self.fusion_transformer_m = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fusion_dim_m, nhead=nhead_m, batch_first=True),
            num_layers=1
        )
        self.num_dynamic_layers_m = 3
        self.fusion_layers_m = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=fusion_dim_m, nhead=nhead_m, batch_first=True, dropout=0.1)
            for _ in range(self.num_dynamic_layers_m)
        ])
        self.layer_weights_m = nn.Parameter(torch.ones(self.num_dynamic_layers_m))
        self.mha_microbe = MultiheadAttention(embed_dim=fusion_dim_m, num_heads=nhead_m, batch_first=True, dropout=0.2)
        self.gated_attn_microbe = GatedAttention(fusion_dim_m)
        self.dynamic_comb_attn_microbe = DynamicCombinationAttention(num_inputs=3, input_dim=fusion_dim_m)

    # def forward(self, x, edge_index, edge_weight=None):
    #     # ----------- 1. 如果是 6 个特征的元组 -----------
    #     if isinstance(x, (tuple, list)):
    #         drug_fg, drug_features,drug_bert, microbe_features,microbe_bert,microbe_path= x  # x 是 (drug_feat, drug_features,drug_bert,microbe_features,microbe_bert,microbe_path) 元组
    #
    #         # if (self.dataset_name == 'aBiofilm'):#aBiofilm要单独处理，所有的特征都要通过mlp变成1160
    #         #     drug_features = self.aBiofilm_drug_mlp(drug_features)
    #         #     drug_fg_reduced = self.aBiofilm_drug_mlp_1(drug_fg)
    #         #     drug_bert = self.aBiofilm_drug_mlp(drug_bert)
    #         if (self.dataset_name == 'MDAD'):#MDAD要单独处理，所有的特征都要通过mlp变成1372
    #             drug_features = self.MDAD_drug_mlp(drug_features)
    #             drug_fg_reduced = self.MDAD_drug_mlp_1(drug_fg)
    #             drug_bert = self.MDAD_drug_mlp(drug_bert)
    #         else:
    #         # # 药物特征降维
    #         #drug_fg_reduced = self.mlp(drug_fg)
    #             drug_fg_reduced = self.mlp(drug_fg)
    #
    #         device = drug_fg_reduced.device
    #
    #         #对微生物的降维处理
    #         if(self.dataset_name=='MDAD'):
    #             microbe_features = self.MDAD_microbe_mlp(microbe_features)
    #             microbe_bert = self.MDAD_microbe_mlp(microbe_bert)
    #             microbe_path = self.MDAD_microbe_mlp(microbe_path)
    #
    #         # if (self.dataset_name == 'DrugVirus'):
    #         #     #print("drug_features shape before MLP:", drug_features.shape)
    #         #     # DrugVirus特征降维处理
    #         #     drug_features = self.DrugVirus_drug_mlp(drug_features)
    #         #     drug_fg_reduced = self.DrugVirus_drug_mlp_1(drug_fg)
    #         #     drug_bert = self.DrugVirus_drug_mlp(drug_bert)
    #         #
    #         #     # 微生物特征处理
    #         #     microbe_features = self.DrugVirus_microbe_mlp(microbe_features)
    #         #     microbe_bert = self.DrugVirus_microbe_mlp(microbe_bert)
    #         #     microbe_path = self.DrugVirus_microbe_mlp(microbe_path)
    #
    #         # 所有输入都保证为 float32 tensor 且在 device 上
    #         drug_features = ensure_tensor_on_device(drug_features, device)
    #         drug_bert = ensure_tensor_on_device(drug_bert, device)
    #
    #         microbe_features = ensure_tensor_on_device(microbe_features, device)
    #         microbe_bert = ensure_tensor_on_device(microbe_bert, device)
    #         microbe_path = ensure_tensor_on_device(microbe_path, device)
    #
    #         # ---- 保证 microbe_feat 也是 tensor 且在 device 上（可选）----
    #         if not torch.is_tensor(microbe_features):
    #             microbe_feat = torch.tensor(microbe_features, dtype=torch.float32, device=device)
    #         else:
    #             microbe_feat = microbe_features.clone().detach().to(dtype=torch.float32, device=device)
    #         # 三个特征逐位相加
    #         #drug_sum = drug_fg_reduced + drug_features + drug_bert
    #
    #
    #
    #         # ======= 微生物特征融合=======(对药物的)
    #             # [n_drug, dim]
    #         drug_fg_w = self.alpha1 * drug_fg_reduced
    #         drug_features_w = self.alpha2 * drug_features
    #         drug_bert_w = self.alpha3 * drug_bert
    #         # 拼成 [n_drug, 3, dim]，3表示三种特征
    #         drug_stack = torch.stack([drug_fg_w, drug_features_w, drug_bert_w], dim=1)  # [n_drug, 3, dim]
    #         # transformer编码，输出 [n_drug, 3, dim]
    #         # drug_fused = self.fusion_transformer(drug_stack)  # [n_drug, 3, dim]
    #         # # 融合成一个向量（比如取第一个token，也可mean/max）
    #         # # 常用做法：mean pooling
    #         # drug_sum = drug_fused.mean(dim=1)  # [n_drug, dim]
    #
    #
    #         # ===== 修改：动态层注意力计算（药物）=====
    #         if self.f == 1:
    #             # ------ 原动态层注意力 ------
    #             layer_outputs = []
    #             x_in = drug_stack
    #             for layer in self.fusion_layers:
    #                 x_out = layer(x_in)
    #                 layer_outputs.append(x_out)
    #                 x_in = x_out
    #             stacked_outputs = torch.stack(layer_outputs, dim=0)
    #             attention_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
    #             drug_fused = torch.sum(stacked_outputs * attention_weights, dim=0)
    #             drug_sum = drug_fused.mean(dim=1)
    #         elif self.f == 2:
    #             # ------ 新增：门控注意力 ------
    #             drug_sum, drug_attn_weights = self.gated_attn_drug(drug_stack)
    #         elif self.f == 3:
    #             # 动态可组合注意力
    #             drug_sum, drug_attn_weights = self.dynamic_comb_attn_drug(drug_stack)
    #         elif self.f == 4:
    #             # 药物特征融合
    #             drug_sum = self.drug_token_fusion([drug_fg, drug_features, drug_bert])
    #
    #         else:#f=0
    #             # ------ 新增：MHA多头注意力 ------
    #             # MultiheadAttention 期望 [batch, seq, embed_dim]
    #             # 这里 batch = n_drug，seq = 3，embed_dim = dim
    #             # Q=K=V=drug_stack
    #             # MHA输出: (n_drug, 3, dim)
    #             attn_output, _ = self.mha_drug(drug_stack, drug_stack, drug_stack)
    #             drug_sum = attn_output.mean(dim=1)  # [n_drug, dim]
    #
    #
    #
    #
    #
    #         # ======= 微生物特征融合 =======（对微生物的）
    #         # [n_drug, dim]
    #         microbe_features_w = self.alpha1_m * microbe_features
    #         microbe_bert_w = self.alpha2_m * microbe_bert
    #         microbe_path_w = self.alpha3_m * microbe_path
    #         # 拼成 [n_drug, 3, dim]，3表示三种特征
    #         microbe_stack = torch.stack([microbe_features_w, microbe_bert_w, microbe_path_w], dim=1)  # [n_drug, 3, dim]
    #         # transformer编码，输出 [n_drug, 3, dim]
    #         # microbe_fused = self.fusion_transformer_m(microbe_stack)  # [n_drug, 3, dim]
    #         # # 融合成一个向量（比如取第一个token，也可mean/max）
    #         # # 常用做法：mean pooling
    #         # microbe_sum = microbe_fused.mean(dim=1)  # [n_drug, dim]
    #
    #         # ===== 修改：动态层注意力计算（微生物）=====
    #         if self.f == 1:
    #             layer_outputs_m = []
    #             x_in_m = microbe_stack
    #             for layer in self.fusion_layers_m:
    #                 x_out_m = layer(x_in_m)
    #                 layer_outputs_m.append(x_out_m)
    #                 x_in_m = x_out_m
    #             stacked_outputs_m = torch.stack(layer_outputs_m, dim=0)
    #             attention_weights_m = F.softmax(self.layer_weights_m, dim=0).view(-1, 1, 1, 1)
    #             microbe_fused = torch.sum(stacked_outputs_m * attention_weights_m, dim=0)
    #             microbe_sum = microbe_fused.mean(dim=1)
    #         elif self.f == 2:
    #             microbe_sum, microbe_attn_weights = self.gated_attn_microbe(microbe_stack)
    #         elif self.f == 3:
    #             microbe_sum, microbe_attn_weights = self.dynamic_comb_attn_microbe(microbe_stack)
    #         elif self.f==4:
    #             # 微生物特征融合
    #             microbe_sum = self.microbe_token_fusion([microbe_features, microbe_bert, microbe_path])
    #         else:
    #             attn_output_m, _ = self.mha_microbe(microbe_stack, microbe_stack, microbe_stack)
    #             microbe_sum = attn_output_m.mean(dim=1)
    #
    #
    #
    #
    #         # 微生物特征降维（如果有microbe_mlp）
    #         if self.use_microbe_mlp and self.microbe_mlp is not None:
    #             microbe_feat_reduced = self.microbe_mlp(microbe_feat)
    #         else:
    #             microbe_feat_reduced = microbe_feat
    #         # 拼接成GCN输入
    #         # 注意：这里建议直接用 torch 操作，不要转 numpy
    #         n_drug = drug_sum.shape[0]
    #         n_microbe = microbe_sum.shape[0]
    #         zero_drug = torch.zeros((n_drug, microbe_sum.shape[1]), device=drug_fg.device)
    #         zero_microbe = torch.zeros((n_microbe, drug_sum.shape[1]), device=drug_fg.device)
    #         #top = torch.cat([zero_drug, drug_sum], dim=1)
    #         #bottom = torch.cat([microbe_sum, zero_microbe], dim=1)
    #         top = torch.cat([drug_sum,zero_drug], dim=1)
    #         bottom = torch.cat([zero_microbe,microbe_sum], dim=1)
    #         X = torch.cat([top, bottom], dim=0)
    #
    #         #特征归一化
    #
    #         # ======= 新增：对每一列做标准化 =======
    #         mean = X.mean(dim=0, keepdim=True)
    #         std = X.std(dim=0, keepdim=True) + 1e-6  # 防止除以0
    #         X = (X - mean) / std
    #         # ====================================
    #
    #         # 送入GCN
    #         embeddings = self.gcn(X, edge_index, edge_weight=edge_weight)
    #         return embeddings, X  # <--- 新增
    #         # ----------- 2. 如果已经是拼接后的大特征矩阵 -----------
    #     else:
    #         # x 已经是 [num_nodes, feat_dim]，直接送入GCN
    #         embeddings = self.gcn(x, edge_index, edge_weight=edge_weight)
    #         return embeddings, x
    def forward(self, x, edge_index, edge_weight=None):
        # ----------- 1. 如果是 6 个特征的元组 -----------
        if isinstance(x, (tuple, list)):
            drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path = x  # x 是 (drug_feat, drug_features,drug_bert,microbe_features,microbe_bert,microbe_path) 元组

            if (self.dataset_name == 'aBiofilm'):#aBiofilm要单独处理，所有的特征都要通过mlp变成1160
                drug_features = self.aBiofilm_drug_mlp(drug_features)
                drug_fg_reduced = self.aBiofilm_drug_mlp_1(drug_fg)
                drug_bert = self.aBiofilm_drug_mlp(drug_bert)
                microbe_features = self.aBiofilm_microbe_mlp(microbe_features)
                microbe_bert = self.aBiofilm_microbe_mlp(microbe_bert)
                microbe_path = self.aBiofilm_microbe_mlp(microbe_path)


            if (self.dataset_name == 'MDAD'):  # MDAD要单独处理，所有的特征都要通过mlp变成1372
                drug_features = self.MDAD_drug_mlp(drug_features)
                drug_fg_reduced = self.MDAD_drug_mlp_1(drug_fg)
                drug_bert = self.MDAD_drug_mlp(drug_bert)

                microbe_features = self.MDAD_microbe_mlp(microbe_features)
                microbe_bert = self.MDAD_microbe_mlp(microbe_bert)
                microbe_path = self.MDAD_microbe_mlp(microbe_path)



            # 对微生物的降维处理
            # if (self.dataset_name == 'MDAD'):
            #     microbe_features = self.MDAD_microbe_mlp(microbe_features)
            #     microbe_bert = self.MDAD_microbe_mlp(microbe_bert)
            #     microbe_path = self.MDAD_microbe_mlp(microbe_path)



            if (self.dataset_name == 'DrugVirus'):
                #print("drug_features shape before MLP:", drug_features.shape)
                # DrugVirus特征降维处理
                drug_features = self.DrugVirus_drug_mlp(drug_features)
                drug_fg_reduced = self.DrugVirus_drug_mlp_1(drug_fg)
                drug_bert = self.DrugVirus_drug_mlp(drug_bert)

                # 微生物特征处理
                microbe_features = self.DrugVirus_microbe_mlp(microbe_features)
                microbe_bert = self.DrugVirus_microbe_mlp(microbe_bert)
                microbe_path = self.DrugVirus_microbe_mlp(microbe_path)
            device = drug_fg_reduced.device
            def _zscore(x):
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, keepdim=True) + 1e-6
                return (x - mean) / std

            if self.dataset_name == 'DrugVirus' or self.dataset_name == 'MDAD' or self.dataset_name == 'aBiofilm':
                drug_fg_reduced = _zscore(drug_fg_reduced)
                drug_features = _zscore(drug_features)
                drug_bert = _zscore(drug_bert)

                microbe_features = _zscore(microbe_features)
                microbe_bert = _zscore(microbe_bert)
                microbe_path = _zscore(microbe_path)


            # 所有输入都保证为 float32 tensor 且在 device 上
            drug_features = ensure_tensor_on_device(drug_features, device)
            drug_bert = ensure_tensor_on_device(drug_bert, device)

            microbe_features = ensure_tensor_on_device(microbe_features, device)
            microbe_bert = ensure_tensor_on_device(microbe_bert, device)
            microbe_path = ensure_tensor_on_device(microbe_path, device)

            # ---- 保证 microbe_feat 也是 tensor 且在 device 上（可选）----
            if not torch.is_tensor(microbe_features):
                microbe_feat = torch.tensor(microbe_features, dtype=torch.float32, device=device)
            else:
                microbe_feat = microbe_features.clone().detach().to(dtype=torch.float32, device=device)
            # 三个特征逐位相加
            # drug_sum = drug_fg_reduced + drug_features + drug_bert

            # ======= 微生物特征融合=======(对药物的)
            # [n_drug, dim]
            drug_fg_w = self.alpha1 * drug_fg_reduced
            drug_features_w = self.alpha2 * drug_features
            drug_bert_w = self.alpha3 * drug_bert
            # 拼成 [n_drug, 3, dim]，3表示三种特征
            drug_stack = torch.stack([drug_fg_w, drug_features_w, drug_bert_w], dim=1)  # [n_drug, 3, dim]
            # transformer编码，输出 [n_drug, 3, dim]
            # drug_fused = self.fusion_transformer(drug_stack)  # [n_drug, 3, dim]
            # # 融合成一个向量（比如取第一个token，也可mean/max）
            # # 常用做法：mean pooling
            # drug_sum = drug_fused.mean(dim=1)  # [n_drug, dim]

            # ===== 修改：动态层注意力计算（药物）=====
            if self.f == 1:
                # ------ 原动态层注意力 ------
                layer_outputs = []
                x_in = drug_stack
                for layer in self.fusion_layers:
                    x_out = layer(x_in)
                    layer_outputs.append(x_out)
                    x_in = x_out
                stacked_outputs = torch.stack(layer_outputs, dim=0)
                attention_weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
                drug_fused = torch.sum(stacked_outputs * attention_weights, dim=0)
                drug_sum = drug_fused.mean(dim=1)
            elif self.f == 2:
                # ------ 新增：门控注意力 ------
                drug_sum, drug_attn_weights = self.gated_attn_drug(drug_stack)
            elif self.f == 3:
                # 动态可组合注意力
                drug_sum, drug_attn_weights = self.dynamic_comb_attn_drug(drug_stack)
            elif self.f == 4:
                # 药物特征融合
                drug_sum = self.drug_token_fusion([drug_fg, drug_features, drug_bert])

            else:  # f=0
                # ------ 新增：MHA多头注意力 ------
                # MultiheadAttention 期望 [batch, seq, embed_dim]
                # 这里 batch = n_drug，seq = 3，embed_dim = dim
                # Q=K=V=drug_stack
                # MHA输出: (n_drug, 3, dim)
                attn_output, _ = self.mha_drug(drug_stack, drug_stack, drug_stack)
                drug_sum = attn_output.mean(dim=1)  # [n_drug, dim]

            # ======= 微生物特征融合 =======（对微生物的）
            # [n_drug, dim]
            microbe_features_w = self.alpha1_m * microbe_features
            microbe_bert_w = self.alpha2_m * microbe_bert
            microbe_path_w = self.alpha3_m * microbe_path
            # 拼成 [n_drug, 3, dim]，3表示三种特征
            microbe_stack = torch.stack([microbe_features_w, microbe_bert_w, microbe_path_w], dim=1)  # [n_drug, 3, dim]
            # transformer编码，输出 [n_drug, 3, dim]
            # microbe_fused = self.fusion_transformer_m(microbe_stack)  # [n_drug, 3, dim]
            # # 融合成一个向量（比如取第一个token，也可mean/max）
            # # 常用做法：mean pooling
            # microbe_sum = microbe_fused.mean(dim=1)  # [n_drug, dim]

            # ===== 修改：动态层注意力计算（微生物）=====
            if self.f == 1:
                layer_outputs_m = []
                x_in_m = microbe_stack
                for layer in self.fusion_layers_m:
                    x_out_m = layer(x_in_m)
                    layer_outputs_m.append(x_out_m)
                    x_in_m = x_out_m
                stacked_outputs_m = torch.stack(layer_outputs_m, dim=0)
                attention_weights_m = F.softmax(self.layer_weights_m, dim=0).view(-1, 1, 1, 1)
                microbe_fused = torch.sum(stacked_outputs_m * attention_weights_m, dim=0)
                microbe_sum = microbe_fused.mean(dim=1)
            elif self.f == 2:
                microbe_sum, microbe_attn_weights = self.gated_attn_microbe(microbe_stack)
            elif self.f == 3:
                microbe_sum, microbe_attn_weights = self.dynamic_comb_attn_microbe(microbe_stack)
            elif self.f == 4:
                # 微生物特征融合
                microbe_sum = self.microbe_token_fusion([microbe_features, microbe_bert, microbe_path])
            else:
                attn_output_m, _ = self.mha_microbe(microbe_stack, microbe_stack, microbe_stack)
                microbe_sum = attn_output_m.mean(dim=1)

            # 微生物特征降维（如果有microbe_mlp）
            if self.use_microbe_mlp and self.microbe_mlp is not None:
                microbe_feat_reduced = self.microbe_mlp(microbe_feat)
            else:
                microbe_feat_reduced = microbe_feat
            # 拼接成GCN输入
            # 注意：这里建议直接用 torch 操作，不要转 numpy
            n_drug = drug_sum.shape[0]
            n_microbe = microbe_sum.shape[0]
            zero_drug = torch.zeros((n_drug, microbe_sum.shape[1]), device=drug_fg.device)
            zero_microbe = torch.zeros((n_microbe, drug_sum.shape[1]), device=drug_fg.device)
            # top = torch.cat([zero_drug, drug_sum], dim=1)
            # bottom = torch.cat([microbe_sum, zero_microbe], dim=1)
            top = torch.cat([drug_sum, zero_drug], dim=1)
            bottom = torch.cat([zero_microbe, microbe_sum], dim=1)
            X = torch.cat([top, bottom], dim=0)

            # 特征归一化

            # ======= 新增：对每一列做标准化 =======
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, keepdim=True) + 1e-6  # 防止除以0
            X = (X - mean) / std
            # ====================================

            # 送入GCN
            #embeddings = self.gcn(X, edge_index, edge_weight=edge_weight)
            # 【【【修改这里：根据gnn_choice调用不同模型】】】
            if self.gnn_choice in [0, 1]:  # 0 for GCN, 1 for GAT
                embeddings = self.gcn(X, edge_index, edge_weight=edge_weight)
            elif self.gnn_choice == 2:  # 2 for MLP
                # MLP模型不使用图结构 (edge_index, edge_weight)，只处理特征矩阵X
                embeddings = self.gcn(X)


            # ==================== 【方案二：潜在空间缩放】 ====================
            # 如果 scaling_factor > 0, 则对GCN输出的嵌入执行L2归一化并缩放

            if self.gnn_choice==0:
                if self.scaling_factor > 0:
                    embeddings = F.normalize(embeddings, p=2, dim=1) * self.scaling_factor
                # ================================================================

            return embeddings, X

        # ----------- 2. 如果已经是拼接后的大特征矩阵 -----------
        else:
            # x 已经是 [num_nodes, feat_dim]，直接送入GCN
            #embeddings = self.gcn(x, edge_index, edge_weight=edge_weight)
            # 【【【修改这里：根据gnn_choice调用不同模型】】】
            if self.gnn_choice in [0, 1]:  # 0 for GCN, 1 for GAT
                embeddings = self.gcn(x, edge_index, edge_weight=edge_weight)
            elif self.gnn_choice == 2:  # 2 for MLP
                # MLP模型不使用图结构 (edge_index, edge_weight)，只处理特征矩阵X
                embeddings = self.gcn(x)


            # ==================== 【方案二：潜在空间缩放】 ====================
            # 在这个分支也需要加上同样的逻辑
            if self.scaling_factor > 0:
                embeddings = F.normalize(embeddings, p=2, dim=1) * self.scaling_factor
            # ================================================================

            return embeddings, x


# gcn_model.py

# ... (你其他的类如 FeatureMLP, GCN_Advanced 等保持不变) ...

def ensure_tensor_on_device(x, device):
    if not torch.is_tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device)
    else:
        return x.clone().detach().to(dtype=torch.float32, device=device)



# ==========================================================
#  请将以下代码添加到 gcn_model.py 文件末尾
# ==========================================================
#你需要定义一个包装类，让 GNNExplainer 能直接拿到“边的预测分数”：
# gcn_model.py
# class GCNWithDecoderWrapper(torch.nn.Module):
#     def __init__(self, gcn_model, decoder, microbe_offset):
#         super().__init__()
#         self.gcn = gcn_model
#         self.decoder = decoder
#         self.microbe_offset = microbe_offset
#
#     def forward(self, x, edge_index, edge_weight=None):
#         embeddings, _ = self.gcn(x, edge_index, edge_weight=edge_weight)
#         src = edge_index[0]
#         dst = edge_index[1]
#         drug_emb = embeddings[src]
#         microbe_emb = embeddings[dst]
#         out = self.decoder(drug_emb, microbe_emb)
#         return out  # 不 squeeze
class GCNWithDecoderWrapper(torch.nn.Module):
    def __init__(self, gcn_model, decoder, microbe_offset):
        super().__init__()
        self.gcn = gcn_model
        self.decoder = decoder
        self.microbe_offset = microbe_offset

    def forward(self, x, edge_index, edge_weight=None):
        # 1. 使用 GCN 模型获取所有节点的嵌入
        # 注意：这里假设 gcn_model 的 forward 返回 (embeddings, X)
        # 如果你的 gcn_model 只返回 embeddings，请使用 embeddings = self.gcn(...)
        embeddings, _ = self.gcn(x, edge_index, edge_weight=edge_weight)

        # 2. 根据 edge_index 提取源节点和目标节点的嵌入
        src_emb = embeddings[edge_index[0]]
        dst_emb = embeddings[edge_index[1]]

        # 3. 【核心修复】确保传递给解码器的张量是 2D 的
        # GNNExplainer 内部可能会用单条边调用模型，导致嵌入是 1D 张量。
        # 而 MLPDecoder 中的 BatchNorm1d 需要至少 2D 的输入 [batch_size, features]。
        if src_emb.dim() == 1:
            src_emb = src_emb.unsqueeze(0)  # 从 [features] 变为 [1, features]
        if dst_emb.dim() == 1:
            dst_emb = dst_emb.unsqueeze(0)  # 从 [features] 变为 [1, features]

        # 4. 将处理好的嵌入传入解码器
        out = self.decoder(src_emb, dst_emb)

        # GNNExplainer期望的输出形状是 [num_edges]，你的decoder输出已经是这个形状了
        return out


class GCNWithDecoderWrapper_cam(torch.nn.Module):
    def __init__(self, gcn_model, decoder, microbe_offset):
        super().__init__()
        self.gcn = gcn_model
        self.decoder = decoder
        self.microbe_offset = microbe_offset

    def forward(self, x, edge_index, edge_weight=None, index=None):
        embeddings, _ = self.gcn(x, edge_index, edge_weight)

        if index is None:
            # 根据需要返回一个 Tensor，这里随便给个0，保证是Tensor
            return torch.zeros(embeddings.size(0), device=embeddings.device)

        src, dst = index

        src_emb = embeddings[src]
        dst_emb = embeddings[dst]

        # **这里做关键改动**：如果是 1D，升成 2D
        if src_emb.dim() == 1:
            src_emb = src_emb.unsqueeze(0)
        if dst_emb.dim() == 1:
            dst_emb = dst_emb.unsqueeze(0)

        out = self.decoder(src_emb, dst_emb)

        if isinstance(out, tuple):
            out = out[0]
        if not torch.is_tensor(out):
            out = torch.as_tensor(out, device=src_emb.device)
        out = out.float()

        if out.dim() == 0:
            out = out.view(1)

        return out




def explain_with_full_graph(explainer, X, edge_index, edge_weight, src, dst, edge_labels):
    # 找到目标边的下标
    edge_indices = (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())
    edge_idx = None
    for i, (s, d) in enumerate(zip(*edge_indices)):
        if s == src and d == dst:
            edge_idx = i
            break
    if edge_idx is None:
        raise ValueError("要解释的边不在 edge_index 里！")
    explanation = explainer(
        x=X,
        edge_index=edge_index,
        edge_weight=edge_weight,
        target=edge_labels,  # 全部标签
        #target=int(edge_labels[edge_idx].item()),
        index=edge_idx
    )
    return explanation, edge_index





import torch
import torch.nn as nn

class FeatureTokenFusion(nn.Module):
    def __init__(self, input_dims, embed_dim, n_heads=4, n_layers=1, output_dim=128):
        """
        input_dims: list，每种特征的输入维度（如[32, 256, 768, 16]）
        embed_dim: 统一嵌入维度，所有特征token都映射成这个维度
        n_heads: Transformer多头数量
        n_layers: Transformer层数
        output_dim: 降维后的输出特征维度
        """
        super().__init__()
        self.num_tokens = len(input_dims)
        self.embeds = nn.ModuleList([nn.Linear(dim, embed_dim) for dim in input_dims])
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.reduce = nn.Linear(embed_dim * self.num_tokens, output_dim)  # 拼接后降维
        self.dropout = nn.Dropout(0.2)  # 0.1~0.5 都可以试试

    def forward(self, feature_list):
        """
        feature_list: list，每个元素是一个 shape [batch_size, dim] 的特征张量
        """
        # 1. 先映射到统一 embed_dim
        token_embeds = []
        for i, feat in enumerate(feature_list):
            # shape: [batch, embed_dim]
            token_embeds.append(self.embeds[i](feat).unsqueeze(1))  # [batch, 1, embed_dim]
        # 2. 拼成一个token序列
        tokens = torch.cat(token_embeds, dim=1)  # [batch, num_tokens, embed_dim]
        # 3. transformer处理
        out = self.transformer(tokens)           # [batch, num_tokens, embed_dim]
        # 4. 展开+降维
        out = out.reshape(out.size(0), -1)       # [batch, num_tokens*embed_dim]
        out = self.reduce(out)                   # [batch, output_dim]

        return out
























def explain_with_khop_subgraph(explainer, X, edge_index, edge_weight, src, dst, num_hops=2):

    import numpy as np
    from torch_geometric.utils import k_hop_subgraph
    import torch

    # 1. 获取k跳子图
    node_idx = [src, dst]
    subset, _, mapping, _ = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=X.shape[0],
        flow='source_to_target'
    )
    subset = subset.cpu().numpy() if hasattr(subset, "cpu") else subset

    # 2. 找到src和dst在子图中的新编号
    src_sub = np.where(subset == src)[0][0]
    dst_sub = np.where(subset == dst)[0][0]

    # 3. 只保留A/B出发且终点在2跳邻居的边
    src_nodes = [src, dst]
    src_edges = edge_index[0].cpu().numpy() if hasattr(edge_index, "cpu") else edge_index[0]
    dst_edges = edge_index[1].cpu().numpy() if hasattr(edge_index, "cpu") else edge_index[1]
    mask = ((np.isin(src_edges, src_nodes)) & (np.isin(dst_edges, subset)))
    selected_edge_indices = np.where(mask)[0]
    sub_edge_index = edge_index[:, selected_edge_indices]
    edge_weight_sub = edge_weight[selected_edge_indices]
    X_sub = X[subset]

    # 4. 找目标边在子图中的编号
    edge_idx_sub = None
    for i in range(sub_edge_index.shape[1]):
        if (sub_edge_index[0, i] == src_sub) and (sub_edge_index[1, i] == dst_sub):
            edge_idx_sub = i
            break
    assert edge_idx_sub is not None, "目标边不在k跳子图内"

    # 5. 去掉目标边
    mask2 = np.ones(sub_edge_index.shape[1], dtype=bool)
    mask2[edge_idx_sub] = False
    sub_edge_index_masked = sub_edge_index[:, mask2]
    edge_weight_sub_masked = edge_weight_sub[mask2]

    # 6. 调用 explainer
    explanation = explainer(
        x=X_sub,
        edge_index=sub_edge_index_masked,
        edge_weight=edge_weight_sub_masked,
        target=torch.tensor([1.0], device=X.device),
        index=0
    )
    return explanation, sub_edge_index_masked, subset

# gcn_model.py 新增

from torch_geometric.explain.algorithm import PGExplainer


import torch
import numpy as np

# def explain_with_pgexplainer(
#     model, decoder, X, edge_index, Sd, Sm, I, microbe_offset, args,
#     drug_fg_norm, drug_features_norm, drug_bert_norm,
#     microbe_features_norm, microbe_bert_norm, microbe_path_norm,
#     test_data, device
# ):
#     """
#     直接复用主流程的数据，无需重新导入。
#     返回：auc_fused_pg, aupr_fused_pg, importance_matrix
#     """
#     import torch_geometric
#     from torch_geometric.explain.config import ModelConfig
#
#     print("PyG version:", torch_geometric.__version__)
#     print("ModelConfig:", ModelConfig)
#     print("ModelConfig.__init__ args:", ModelConfig.__init__.__code__.co_varnames)
#     print("ModelConfig.__init__ defaults:", ModelConfig.__init__.__defaults__)
#
#
#
#     from torch_geometric.explain.config import ModelConfig, ModelMode, ModelTaskLevel, ModelReturnType
#
#     # 正确写法（适用于PyG 2.6.1）：
#     from torch_geometric.explain.config import ModelReturnType
#
#     pgexplainer.connect(
#         model_config={
#             'mode': 'binary_classification',
#             'task_level': 'edge',
#             'return_type': 'raw'
#         },
#         model=wrapped_model
#     )
#
#     print(model_config)
#     print(type(model_config.mode))
#     print(type(model_config.task_level))
#     print(type(model_config.return_type))
#
#
#
#     # 1. 包装模型
#     # 1. 包装模型
#     wrapped_model = GCNWithDecoderWrapper(model, decoder, microbe_offset).to(device)
#     wrapped_model.eval()
#
#     # 2. 初始化PGExplainer
#     pgexplainer = PGExplainer(wrapped_model, channels=args.hidden_dim, num_hops=2).to(device)
#     optimizer_pg = torch.optim.Adam(pgexplainer.parameters(), lr=0.01)
#
#     # 3. 正确connect（只需要一次！）
#     pgexplainer.connect(
#         model_config={
#             'mode': 'binary_classification',
#             'task_level': 'edge',
#             'return_type': 'raw'
#         },
#         model=wrapped_model
#     )
#
#     num_train_nodes = X.shape[0]
#     # 1. 构造 edge_labels
#     row_np = edge_index[0].cpu().numpy()
#     col_np = edge_index[1].cpu().numpy()
#     labels = []
#     for src_, dst_ in zip(row_np, col_np):
#         # 判断是否为药物-微生物边
#         if src_ < microbe_offset and dst_ >= microbe_offset:
#             label = int(I[src_, dst_ - microbe_offset] == 1)
#         else:
#             label = 0
#         labels.append(label)
#     edge_labels = torch.tensor(labels, dtype=torch.long, device=X.device)
#
#     # 2. 训练 PGExplainer
#     for epoch_pg in range(10):
#         optimizer_pg.zero_grad()
#         loss = pgexplainer.train(
#             epoch_pg,
#             wrapped_model,
#             X,
#             edge_index,
#             target=edge_labels
#         )
#         loss.backward()
#         optimizer_pg.step()
#         print(f'[PGExplainer] Epoch {epoch_pg}, Loss {loss.item():.4f}')
#
#     # 3. 累计全局边重要性
#     edge_index_np = edge_index.cpu().numpy()
#     importance_matrix = np.zeros((num_train_nodes, num_train_nodes))
#     count_matrix = np.zeros((num_train_nodes, num_train_nodes))
#     pgexplainer.eval()
#     with torch.no_grad():
#         for node_idx in range(num_train_nodes):
#             edge_mask = pgexplainer.explain_node(node_idx, wrapped_model, X, edge_index)
#             edge_mask_np = edge_mask.cpu().numpy()
#             for i, (src, dst) in enumerate(edge_index_np.T):
#                 importance_matrix[src, dst] += edge_mask_np[i]
#                 importance_matrix[dst, src] += edge_mask_np[i]
#                 count_matrix[src, dst] += 1
#                 count_matrix[dst, src] += 1
#     importance_matrix = np.divide(importance_matrix, count_matrix, out=np.zeros_like(importance_matrix), where=count_matrix!=0)
#     if importance_matrix.max() > 0:
#         importance_matrix = (importance_matrix - importance_matrix.min()) / (importance_matrix.max() - importance_matrix.min() + 1e-8)
#
#     # 4. 融合邻接、生成edge_index/edge_weight
#     from data_utils import fuse_adj_with_mask
#     A_fused_pg = fuse_adj_with_mask(Sd, Sm, I, importance_matrix, lambda_d=0.5, lambda_m=0.5, lambda_I=0.5)
#     row_f_pg, col_f_pg = np.where(A_fused_pg != 0)
#     edge_index_f_pg = np.stack([row_f_pg, col_f_pg], axis=0)
#     edge_weight_f_pg = A_fused_pg[row_f_pg, col_f_pg]
#     edge_index_f_pg = torch.tensor(edge_index_f_pg, dtype=torch.long, device=device)
#     edge_weight_f_pg = torch.tensor(edge_weight_f_pg, dtype=torch.float32, device=device)
#
#     # 5. evaluate_gcn评估
#     from train_eval import evaluate_gcn
#     auc_fused_pg, aupr_fused_pg = evaluate_gcn(
#         model, decoder, test_data, edge_index_f_pg, edge_weight_f_pg,
#         drug_fg_norm, drug_features_norm, drug_bert_norm,
#         microbe_features_norm, microbe_bert_norm, microbe_path_norm,
#         microbe_offset, device=device
#     )
#     print(f'【PGExplainer增强邻接矩阵】Test AUC: {auc_fused_pg:.4f}, AUPR: {aupr_fused_pg:.4f}')
#     return auc_fused_pg, aupr_fused_pg, importance_matrix
#
# ==========================================================
#  请将 gcn_model.py 文件中的 explain_with_pgexplainer 函数
#  整体替换为以下最终代码
# ==========================================================
def explain_with_pgexplainer(model, decoder, X, edge_index, edge_weight, Sd, Sm, I, microbe_offset, args,
                             device, top_percent=0.5, lambda_new=0.1, batch_size=1024):
    """
    [方法变更] 由于PGExplainer的限制，此函数现在通过直接的模型预测来增强邻接矩阵。
    它会识别潜在的新关联，使用已训练的GCN模型预测其存在概率，并筛选高分新边。

    Args:
        model: 原始的GCN模型 (GCNWithMLP)。
        decoder: 解码器 (MLPDecoder)。
        X: 节点特征矩阵。
        edge_index: 边的索引 (此函数中未使用，但保留以兼容接口)。
        edge_weight: 边的权重 (此函数中未使用，但保留以兼容接口)。
        Sd, Sm, I: 原始的相似性矩阵和关联矩阵。
        microbe_offset: 微生物节点在全局索引中的偏移量。
        args: 命令行参数。
        device: 'cuda' or 'cpu'。
        top_percent (float): 选择得分最高的潜在边的百分比。
        lambda_new (float): 新增边的权重。
        batch_size (int): 批量预测时每个批次的大小。

    Returns:
        I_fused (np.ndarray): 融合了新边的关联矩阵。
        candidate_scores (np.ndarray): 所有候选边的预测得分。
    """
    import torch
    import numpy as np
    from tqdm import tqdm

    print("===== 方法变更：使用直接预测而非PGExplainer来增强邻接矩阵 =====")

    # 1. 创建一个包装模型，使其输出为边的预测分数
    wrapped_model = GCNWithDecoderWrapper(model, decoder, microbe_offset).to(device)
    wrapped_model.eval()

    num_drug = Sd.shape[0]
    num_microbe = Sm.shape[0]

    # 2. 找出所有当前不存在的“药物-微生物”候选边
    # I矩阵的索引对应的是药物ID和微生物ID
    drug_indices, microbe_indices_in_I = np.where(I == 0)

    # 将微生物ID转换为全局图中的节点ID
    microbe_indices_global = microbe_indices_in_I + num_drug

    # 创建候选边的 edge_index
    candidate_edge_index = torch.tensor(
        np.array([drug_indices, microbe_indices_global]),
        dtype=torch.long
    ).to(device)

    num_candidates = candidate_edge_index.size(1)
    print(f"找到 {num_candidates} 条潜在的候选关联边。")

    # 3. 批量预测候选边的存在概率
    all_scores = []
    print("开始批量预测候选边的分数...")
    with torch.no_grad():
        for i in tqdm(range(0, num_candidates, batch_size)):
            batch_edge_index = candidate_edge_index[:, i:i + batch_size]
            # GCNWithDecoderWrapper的forward需要x, edge_index
            # 这里的edge_index是我们要查询的边，而不是图的结构边
            # 我们需要传入完整的图结构给GCN，然后用decoder查询特定边

            # 首先，通过GCN获取所有节点的嵌入
            node_embeddings = model(X, model.edge_index, model.edge_weight)

            # 然后，用解码器对批量的候选边进行打分
            src_embeds = node_embeddings[batch_edge_index[0]]
            dst_embeds = node_embeddings[batch_edge_index[1]]

            batch_scores = decoder(src_embeds, dst_embeds).squeeze()
            all_scores.append(batch_scores.cpu())

    candidate_scores_tensor = torch.cat(all_scores)
    candidate_scores = candidate_scores_tensor.numpy()

    # 4. 筛选出得分最高的潜在关联
    k = int(num_candidates * top_percent / 100)

    topk_indices = []
    if k > 0 and num_candidates > 0:
        # 使用argpartition可以更高效地找到top-k，而无需完全排序
        # 我们需要最大的k个，所以对负分数找最小的k个
        topk_indices = np.argpartition(-candidate_scores, k)[:k]
        print(f"已筛选出 Top {k} 条新的潜在关联边。")
    else:
        print("未筛选出新的潜在关联边。")

    # 5. 将新边融合到I矩阵中
    I_fused = I.copy()
    for idx in topk_indices:
        src_drug = drug_indices[idx]
        dst_microbe_in_I = microbe_indices_in_I[idx]
        I_fused[src_drug, dst_microbe_in_I] = lambda_new

    # 6. 返回增强后的I矩阵和所有候选边的得分
    # 注意：这里的scores只对应候选边，而不是图中所有边
    return I_fused, candidate_scores




import torch
from torch.autograd import Function
import torch.nn as nn

# 1. 定义梯度反转层 (Gradient Reversal Layer)
# 这个模块在正向传播时什么都不做，但在反向传播时会把梯度乘以一个负的常数。
class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        # 保存lambda_val以便在反向传播时使用
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 这是GRL的核心：反转梯度
        # grad_output是来自后续层的梯度，我们将其乘以-lambda_val
        # 第二个返回值对应forward的第二个输入lambda_val，它不需要梯度
        return (grad_output.neg() * ctx.lambda_val), None

class GradientReverse(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReverse, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        # 应用梯度反转功能
        return GradientReverseFunction.apply(x, self.lambda_val)

# 2. 定义对抗判别器 (Adversarial Discriminator)
# 这是一个简单的多层感知机（MLP），用于区分特征来自哪个任务。
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # 输入节点嵌入，输出一个logit值，表示其属于某个领域的概率
        return self.network(x)
