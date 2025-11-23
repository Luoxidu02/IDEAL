import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
#from gcn_model import GCN
import numpy as np
import time
import copy # <--- 在这里添加
from gcn_model import GradientReverse, Discriminator
from torch.cuda.amp import autocast, GradScaler

from gcn_model import  MLPDecoder

# train_eval.py
from gcn_model import GCNWithMLP, MLPDecoder
import numpy as np
import os                # <<-- 添加这一行
import matplotlib.pyplot as plt

# 在微调阶段，对齐MLP将训练
# def train_alignment_mlps(
#     drugvirus_feats, mdad_feats, device, epochs=100, lr=0.001
# ):
#     mlp_list = []
#     optim_params = []
#     for dv_feat, mdad_feat in zip(drugvirus_feats, mdad_feats):
#         mlp = torch.nn.Sequential(
#             torch.nn.Linear(dv_feat.shape[1], mdad_feat.shape[1]),
#             torch.nn.ReLU()
#         ).to(device)
#         mlp_list.append(mlp)
#         optim_params += list(mlp.parameters())
#     optimizer = torch.optim.Adam(optim_params, lr=lr,weight_decay=5e-4)
#     target_means = [torch.tensor(m.mean(axis=0), dtype=torch.float32, device=device) for m in mdad_feats]
#     target_stds = [torch.tensor(m.std(axis=0), dtype=torch.float32, device=device) for m in mdad_feats]
#     dv_tensors = [torch.tensor(f, dtype=torch.float32, device=device) for f in drugvirus_feats]
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         losses = []
#         for i, mlp in enumerate(mlp_list):
#             pred = mlp(dv_tensors[i])
#             loss = ((pred.mean(dim=0) - target_means[i]) ** 2).mean() + \
#                    ((pred.std(dim=0) - target_stds[i]) ** 2).mean()
#             losses.append(loss)
#         total_loss = sum(losses)
#         total_loss.backward()
#         optimizer.step()
#         if (epoch + 1) % 10 == 0:
#             print(f"Align Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}")
#     return mlp_list
#


# def train_alignment_mlps(
#         drugvirus_feats,
#         mdad_feats,
#         device,
#         epochs: int = 200,
#         lr: float = 1e-3,
#         batch_size: int = 256,
# ):
#
#     """
#     训练一组 alignment MLP，将 DrugVirus 特征映射到 MDAD 特征空间。
#     loss = 均值差 + 方差差 + CORAL(协方差差) + 随机配对 L2。
#     """
#     from torch.utils.data import TensorDataset, DataLoader
#
#     def _make_mlp(in_dim, out_dim):
#         mlp = torch.nn.Sequential(
#             torch.nn.Linear(in_dim, 2 * out_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(2 * out_dim, out_dim),
#         ).to(device)
#         torch.nn.init.eye_(mlp[-1].weight)
#         torch.nn.init.zeros_(mlp[-1].bias)
#         return mlp
#
#
#
#     mlp_list, optim_params, dv_loaders = [], [], []
#
#     for dv_feat, mdad_feat in zip(drugvirus_feats, mdad_feats):
#         dv_tensor = torch.tensor(dv_feat, dtype=torch.float32, device=device)
#         mlp = _make_mlp(dv_tensor.size(1), mdad_feat.shape[1])
#         mlp_list.append(mlp)
#         optim_params += list(mlp.parameters())
#
#         # 如果样本数小于batch_size，drop_last=True 会导致loader为空
#         # 改为 drop_last=False 确保至少有一个batch
#         drop_last_flag = dv_tensor.size(0) > batch_size
#         #dv_ds = TensorDataset(dv_tensor)
#         mdad_tensor = torch.tensor(mdad_feat, dtype=torch.float32, device=device)
#         # 确保输入和标签的样本数量一致
#         assert dv_tensor.size(0) == mdad_tensor.size(0), "输入和标签的样本数必须一致"
#
#         # 将输入(X)和标签(Y)打包成对
#         dv_ds = TensorDataset(dv_tensor, mdad_tensor)
#
#         dv_loaders.append(DataLoader(dv_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last_flag))
#
#     optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=5e-4)
#     # 使用标准的均方误差损失函数
#     criterion = torch.nn.MSELoss()
#
#     for epoch in range(epochs):
#         total_epoch_loss = 0.0
#         batch_count = 0
#
#         for loader_idx, dv_loader in enumerate(dv_loaders):
#             # 遍历数据加载器，每次得到一个批次的输入(x)和标签(y)
#             for x_batch, y_batch in dv_loader:
#                 optimizer.zero_grad()
#
#                 mlp = mlp_list[loader_idx]
#
#                 # 模型进行预测
#                 pred = mlp(x_batch)
#
#                 # 直接计算预测值和真实标签之间的MSE损失
#                 loss = criterion(pred, y_batch)
#
#                 loss.backward()
#                 optimizer.step()
#
#                 total_epoch_loss += loss.item()
#                 batch_count += 1
#
#         if (epoch + 1) % 10 == 0:
#             avg_epoch_loss = total_epoch_loss / batch_count if batch_count > 0 else 0.0
#             print(f"[Align] Epoch {epoch + 1:3d}/{epochs}, Avg Regression Loss: {avg_epoch_loss:.5f}")
#
#     return mlp_list


# train_eval.py

# 将这个函数整体替换掉
# def train_alignment_mlps(
#         drugvirus_feats,
#         mdad_feats,
#         device,
#         epochs: int = 200,
#         lr: float = 1e-3,
#         batch_size: int = 256,
# ):
#     """
#     【修改后】: 这个函数现在只创建不对齐的MLP模块，不再进行训练。
#     训练将在主 EWC 循环中进行。
#     """
#
#     def _make_mlp(in_dim, out_dim):
#         mlp = torch.nn.Sequential(
#             torch.nn.Linear(in_dim, (in_dim + out_dim) // 2),  # 使用更平滑的中间维度
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.2),
#             torch.nn.Linear((in_dim + out_dim) // 2, out_dim),
#         ).to(device)
#         # 移除 eye_ 初始化，让网络从随机状态开始学习
#         # torch.nn.init.eye_(mlp[-1].weight)
#         # torch.nn.init.zeros_(mlp[-1].bias)
#         return mlp
#
#     mlp_list = []
#     for dv_feat, mdad_feat in zip(drugvirus_feats, mdad_feats):
#         mlp = _make_mlp(dv_feat.shape[1], mdad_feat.shape[1])
#         mlp_list.append(mlp)
#
#     # 将独立的MLP列表封装成一个nn.ModuleList，这样它们就能被PyTorch的优化器正确识别
#     alignment_mlps = nn.ModuleList(mlp_list).to(device)
#
#     # 移除整个训练循环 (for epoch in ...)，直接返回未训练的MLP模块
#     print(f"成功创建 {len(alignment_mlps)} 个对齐MLP模块（未训练）。")
#     return alignment_mlps
# train_eval.py

# 将这个函数整体替换掉
def train_alignment_mlps(
        drugvirus_feats,
        mdad_feats,
        device,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
):
    """
    【修改后】: 这个函数现在只创建MLP模块，并将其初始化为近似恒等映射。
    它不再进行预训练，因为训练将在主 EWC 循环中进行。
    """

    def _make_mlp(in_dim, out_dim):
        # 必须确保输入和输出维度相同才能进行恒等初始化
        if in_dim != out_dim:
            # 如果维度不同，我们创建一个简单的线性层，但无法做恒等初始化。
            # 这在你的流程中不应该发生，因为外部对齐已经处理了维度问题。
            print(f"警告: 创建对齐MLP时输入({in_dim})和输出({out_dim})维度不同，无法进行恒等初始化。")
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim)
            ).to(device)
        else:
            # 维度相同时，创建单层线性网络
            mlp = torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim)
            ).to(device)
            # 核心：将权重初始化为单位矩阵，偏置初始化为零，实现恒等映射
            print(f"成功: 创建了一个输入输出维度为 {in_dim} 的恒等初始化MLP。")
            torch.nn.init.eye_(mlp[0].weight)
            torch.nn.init.zeros_(mlp[0].bias)

        return mlp

    mlp_list = []
    # 注意：这里的输入特征 drugvirus_feats 实际上是已经经过外部对齐的，
    # 所以它们的维度应该和 mdad_feats 相同。
    for dv_feat, mdad_feat in zip(drugvirus_feats, mdad_feats):
        # 我们使用mdad_feat的维度作为输入和输出维度，因为这是对齐后的目标维度
        dim = mdad_feat.shape[1]
        mlp = _make_mlp(dim, dim)
        mlp_list.append(mlp)

    # 将独立的MLP列表封装成一个nn.ModuleList
    alignment_mlps = nn.ModuleList(mlp_list).to(device)

    print(f"成功创建 {len(alignment_mlps)} 个对齐MLP模块（已初始化为恒等映射）。")
    return alignment_mlps




def build_gcn_features(Fd, Fm):#将微生物与药物特征堆叠
    # n_drug = Fd.shape[0]
    # n_microbe = Fm.shape[0]
    n_drug = Fd.shape[0]
    n_microbe = Fm.shape[0]
    zero_drug = np.zeros((n_drug, Fm.shape[1]))
    zero_microbe = np.zeros((n_microbe, Fd.shape[1]))
    top = np.hstack((zero_drug, Fd))
    bottom = np.hstack((Fm, zero_microbe))
    X = np.vstack((top, bottom))
    return X

# def train_gcn(train_data,  edge_index, edge_weight,drug_fg,drug_features,drug_bert, microbe_features,microbe_bert,microbe_path, microbe_offset,epochs=100, lr=0.01, hidden=64, dropout=0.5,
#               args=None,device='cpu'):
#     model = GCNWithMLP(
#         drug_in_dim=drug_fg.shape[1],
#         drug_out_dim=drug_fg.shape[0],
#         microbe_dim=microbe_features.shape[1],
#         microbe_out_dim=microbe_features.shape[1],  # 传递原始维度
#         gcn_hidden=hidden,
#         dropout=dropout,
#         use_microbe_mlp=False, # 只对药物进行 MLP
#         dataset_name=args.dataset
#     ).to(device)
#
#     decoder = MLPDecoder(hidden).to(device)
#     optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
#     criterion = nn.BCEWithLogitsLoss()
#     model.train()
#     decoder.train()
#
#     # 我们需要保存最后一个epoch的梯度
#     last_epoch_gradients = None
#     # ====== 这三行一定要加上！======
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32).to(device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32).to(device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32).to(device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32).to(device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32).to(device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32).to(device)
#
#     # ============================
#     for epoch in range(epochs):
#         optimizer.zero_grad()# 🌟 清空梯度
#         drug_idx, microbe_idx, labels = train_data
#
#         drug_fg = torch.tensor(drug_fg, dtype=torch.float32).to(device)
#         #microbe_feat = torch.tensor(microbe_features, dtype=torch.float32).to(device)
#
#         #adj = torch.tensor(A, dtype=torch.float32).to(device)
#
#         # 获取GCN嵌入
#         embeddings, X = model((drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#                               edge_index, edge_weight)
#
#         # 【关键修改1】: 告诉PyTorch我们需要计算这个中间变量的梯度
#         #embeddings.requires_grad_(True)
#         embeddings.retain_grad()
#
#         drug_emb = embeddings[drug_idx]
#         microbe_emb = embeddings[microbe_offset + microbe_idx]#原本这里是1180
#         logits = decoder(drug_emb, microbe_emb)
#         loss = criterion(logits, torch.tensor(labels, dtype=torch.float32).to(device))
#
#         loss.backward()
#
#         # 【关键修改2】: 在优化器更新前，保存梯度
#         if epoch == epochs - 1:
#             last_epoch_gradients = embeddings.grad.detach().cpu().numpy()
#
#         optimizer.step()
#
#         if (epoch + 1) % 40 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
#
#     # 【关键修改3】: 返回模型、解码器和梯度
#     return model, decoder, last_epoch_gradients,embeddings,X
#
# def train_gcn(
#     train_data, edge_index, edge_weight,
#     drug_fg, drug_features, drug_bert,
#     microbe_features, microbe_bert, microbe_path, microbe_offset,
#     epochs=100, lr=0.01, hidden=64, dropout=0.5,
#     args=None, device='cpu', batch_size=256  # 新增 batch_size 参数，默认256
# ):
#     model = GCNWithMLP(
#         drug_in_dim=drug_fg.shape[1],
#         drug_out_dim=drug_fg.shape[0],
#         microbe_dim=microbe_features.shape[1],
#         microbe_out_dim=microbe_features.shape[1],  # 传递原始维度
#         gcn_hidden=hidden,
#         dropout=dropout,
#         use_microbe_mlp=False,  # 只对药物进行 MLP
#         dataset_name=args.dataset
#     ).to(device)
#
#     decoder = MLPDecoder(hidden).to(device)
#     optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
#     criterion = nn.BCEWithLogitsLoss()
#     model.train()
#     decoder.train()
#
#     last_epoch_gradients = None
#
#     # ====== 这几行只需转换一次！======
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32).to(device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32).to(device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32).to(device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32).to(device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32).to(device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32).to(device)
#
#     drug_idx, microbe_idx, labels = train_data
#     drug_idx = np.array(drug_idx)
#     microbe_idx = np.array(microbe_idx)
#     labels = np.array(labels)
#
#     num_samples = len(drug_idx)
#
#
#     for epoch in range(epochs):
#         # ⬅️ 新增：每40轮起始时间
#         if epoch % 40 == 0:
#             start_time_40 = time.time()
#
#         model.train()
#         decoder.train()
#         permutation = np.random.permutation(num_samples)
#         total_loss = 0.0
#
#         for i in range(0, num_samples, batch_size):
#             idx = permutation[i:i+batch_size]
#             batch_drug_idx = drug_idx[idx]
#             batch_microbe_idx = microbe_idx[idx]
#             batch_labels = labels[idx]
#
#             optimizer.zero_grad()
#
#             # 获取GCN嵌入（全图特征，batch采样边）
#             embeddings, X = model(
#                 (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#                 edge_index, edge_weight
#             )
#             embeddings.retain_grad()
#
#             drug_emb = embeddings[batch_drug_idx]
#             microbe_emb = embeddings[microbe_offset + batch_microbe_idx]
#             logits = decoder(drug_emb, microbe_emb)
#             loss = criterion(logits, torch.tensor(batch_labels, dtype=torch.float32).to(device))
#             loss.backward()
#
#             # 只保存最后一个epoch最后一个batch的梯度
#             if epoch == epochs - 1 and i + batch_size >= num_samples:
#                 last_epoch_gradients = embeddings.grad.detach().cpu().numpy()
#
#             optimizer.step()
#             total_loss += loss.item()
#
#         if (epoch + 1) % 40 == 0:
#             end_time_40 = time.time()
#             elapsed_40 = end_time_40 - start_time_40
#             avg_loss = total_loss / (num_samples // batch_size + int(num_samples % batch_size != 0))
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, 40-epoch time: {elapsed_40:.2f} sec")
#
#     return model, decoder, last_epoch_gradients, embeddings, X
#

# train_eval.py

import time  # 确保导入了 time 模块

#
# def train_gcn(
#         train_data, edge_index, edge_weight,
#         drug_fg, drug_features, drug_bert,
#         microbe_features, microbe_bert, microbe_path, microbe_offset,
#         epochs=100, lr=0.01, hidden=64, dropout=0.5,
#         args=None, device='cpu', batch_size=256
# ):
#     model = GCNWithMLP(
#         drug_in_dim=drug_fg.shape[1],
#         drug_out_dim=drug_fg.shape[0],
#         microbe_dim=microbe_features.shape[1],
#         microbe_out_dim=microbe_features.shape[1],
#         gcn_hidden=hidden,
#         dropout=dropout,
#         use_microbe_mlp=False,
#         dataset_name=args.dataset
#     ).to(device)
#
#     decoder = MLPDecoder(hidden).to(device)
#     optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
#     criterion = nn.BCEWithLogitsLoss()
#
#     # 特征转换只需要一次
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)
#
#     drug_idx, microbe_idx, labels = train_data
#     drug_idx = np.array(drug_idx)
#     microbe_idx = np.array(microbe_idx)
#     labels = np.array(labels)
#     num_samples = len(drug_idx)
#
#     last_epoch_gradients = None
#     final_embeddings = None
#     final_X = None
#
#     for epoch in range(epochs):
#         if epoch % 40 == 0:
#             start_time_40 = time.time()
#
#         model.train()
#         decoder.train()
#
#         # ======================= 核心优化点 =======================
#         # 在每个 epoch 开始时，计算一次全图的 GCN 嵌入
#         optimizer.zero_grad()
#         embeddings, X = model(
#             (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#             edge_index, edge_weight
#         )
#         embeddings.retain_grad()  # 依然需要保留梯度
#         # =========================================================
#
#         permutation = np.random.permutation(num_samples)
#         total_loss = 0.0
#
#         for i in range(0, num_samples, batch_size):
#             idx = permutation[i:i + batch_size]
#             batch_drug_idx = drug_idx[idx]
#             batch_microbe_idx = microbe_idx[idx]
#             batch_labels = labels[idx]
#
#             # 直接从预先计算好的 embeddings 中取值，不再重新计算GCN
#             drug_emb = embeddings[batch_drug_idx]
#             microbe_emb = embeddings[microbe_offset + batch_microbe_idx]
#
#             logits = decoder(drug_emb, microbe_emb)
#             loss = criterion(logits, torch.tensor(batch_labels, dtype=torch.float32, device=device))
#
#             # 在 batch 循环中累加损失，在 epoch 结束后统一反向传播
#             total_loss += loss
#
#         # 在所有 batch 的损失累加完后，进行一次反向传播和优化
#         total_loss.backward()
#         optimizer.step()
#
#         if epoch == epochs - 1:
#             last_epoch_gradients = embeddings.grad.detach().cpu().numpy() if embeddings.grad is not None else None
#             final_embeddings = embeddings.detach()
#             final_X = X.detach()
#
#         if (epoch + 1) % 40 == 0:
#             end_time_40 = time.time()
#             elapsed_40 = end_time_40 - start_time_40
#             avg_loss = total_loss.item() / (num_samples // batch_size + int(num_samples % batch_size != 0))
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, 40-epoch time: {elapsed_40:.2f} sec")
#
#     return model, decoder, last_epoch_gradients, final_embeddings, final_X
#
# ===================================================================

# =================================================================================
#  请用以下完整函数替换您文件中旧的 train_gcn 函数
# =================================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
# 确保 evaluate_gcn 函数在当前文件中可用，或者已从别处正确导入
# 例如: from .train_eval import evaluate_gcn
from gcn_model import GCNWithMLP, MLPDecoder


# def train_gcn(
#         train_data, edge_index, edge_weight,
#         drug_fg, drug_features, drug_bert,
#         microbe_features, microbe_bert, microbe_path, microbe_offset,
#         epochs=100, lr=0.01, hidden=64, dropout=0.5,
#         args=None, device='cpu', batch_size=256,
#         test_data=None  # <--- 第1处修改：增加 test_data 参数
# ):
#     model = GCNWithMLP(
#         drug_in_dim=drug_fg.shape[1],
#         drug_out_dim=drug_fg.shape[0],
#         microbe_dim=microbe_features.shape[1],
#         microbe_out_dim=microbe_features.shape[1],
#         gcn_hidden=hidden,
#         dropout=dropout,
#         use_microbe_mlp=False,
#         dataset_name=args.dataset
#     ).to(device)
#
#     decoder = MLPDecoder(hidden).to(device)
#     optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
#     criterion = nn.BCEWithLogitsLoss()
#
#     # 特征转换只需要一次
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)
#
#     drug_idx, microbe_idx, labels = train_data
#     drug_idx = np.array(drug_idx)
#     microbe_idx = np.array(microbe_idx)
#     labels = np.array(labels)
#     num_samples = len(drug_idx)
#
#     last_epoch_gradients = None
#     final_embeddings = None
#     final_X = None
#
#     for epoch in range(epochs):
#         if epoch % 40 == 0:
#             start_time_40 = time.time()
#
#         model.train()
#         decoder.train()
#
#         # ======================= 核心优化点 (保持不变) =======================
#         optimizer.zero_grad()
#         embeddings, X = model(
#             (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#             edge_index, edge_weight
#         )
#         embeddings.retain_grad()
#         # ====================================================================
#
#         permutation = np.random.permutation(num_samples)
#         total_loss = 0.0
#
#         for i in range(0, num_samples, batch_size):
#             idx = permutation[i:i + batch_size]
#             batch_drug_idx = drug_idx[idx]
#             batch_microbe_idx = microbe_idx[idx]
#             batch_labels = labels[idx]
#
#             drug_emb = embeddings[batch_drug_idx]
#             microbe_emb = embeddings[microbe_offset + batch_microbe_idx]
#
#             logits = decoder(drug_emb, microbe_emb)
#             loss = criterion(logits, torch.tensor(batch_labels, dtype=torch.float32, device=device))
#
#             total_loss += loss
#
#         total_loss.backward()
#         optimizer.step()
#
#         if epoch == epochs - 1:
#             last_epoch_gradients = embeddings.grad.detach().cpu().numpy() if embeddings.grad is not None else None
#             final_embeddings = embeddings.detach()
#             final_X = X.detach()
#
#         # ======================= 第2处修改：在此处增加评估逻辑 =======================
#         if (epoch + 1) % 40 == 0:
#             end_time_40 = time.time()
#             elapsed_40 = end_time_40 - start_time_40
#             avg_loss = total_loss.item() / (num_samples / batch_size)
#
#             # 先构建基础的输出字符串
#             output_string = f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, 40-epoch time: {elapsed_40:.2f} sec"
#
#             # 如果有测试数据，进行评估并追加结果到字符串
#             if test_data is not None:
#                 model.eval()
#                 decoder.eval()
#
#                 with torch.no_grad():
#                     test_auc, test_aupr = evaluate_gcn(
#                         model, decoder, test_data, edge_index, edge_weight,
#                         drug_fg, drug_features, drug_bert,
#                         microbe_features, microbe_bert, microbe_path,
#                         microbe_offset, device
#                     )
#
#                 # 使用 += 来追加测试结果
#                 output_string += f", Test AUC: {test_auc:.4f}, Test AUPR: {test_aupr:.4f}"
#
#             # 最后，只用一个 print 语句输出所有信息
#             print(output_string)
#
#             # 在这个代码块结束后，下一次循环开始时，
#             # model.train() 和 decoder.train() 会被自动调用，无需手动切换回来。
#         # ========================================================================
#
#     return model, decoder, last_epoch_gradients, final_embeddings, final_X

# train_eval.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from gcn_model import GCNWithMLP, MLPDecoder


# def train_gcn(
#         train_data, edge_index, edge_weight,
#         drug_fg, drug_features, drug_bert,
#         microbe_features, microbe_bert, microbe_path, microbe_offset,
#         epochs=100, lr=0.01, hidden=64, dropout=0.5,
#         args=None, device='cpu', batch_size=256,
#         test_data=None,
#         fold_num=0,
#         save_dir='.',
#         plot_filename=None , # <--- 【核心修改1】: 新增 plot_filename 参数
#         weight_decay = 0.0 , # <--- 【核心修改1：增加参数】
# # --- 【新增】早停相关参数 ---
#         use_early_stopping=False,
#         patience=50
# ):
#     model = GCNWithMLP(
#         drug_in_dim=drug_fg.shape[1],
#         drug_out_dim=drug_fg.shape[0],
#         microbe_dim=microbe_features.shape[1],
#         microbe_out_dim=microbe_features.shape[1],
#         gcn_hidden=hidden,
#         dropout=dropout,
#         use_microbe_mlp=False,
#         dataset_name=args.dataset
#     ).to(device)
#
#     decoder = MLPDecoder(hidden).to(device)
#     optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr,weight_decay=args.wd_retrain )
#     # ======================== 【在这里新增】 ========================
#     # 1. 创建学习率调度器
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer,
#         step_size=args.lr_step_size,
#         gamma=args.lr_gamma
#     )
#     # ===============================================================
#
#     criterion = nn.BCEWithLogitsLoss()
#
#     # 特征转换只需要一次
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)
#
#     drug_idx, microbe_idx, labels = train_data
#     drug_idx = np.array(drug_idx)
#     microbe_idx = np.array(microbe_idx)
#     labels = np.array(labels)
#     num_samples = len(drug_idx)
#
#     last_epoch_gradients = None
#     final_embeddings = None
#     final_X = None
#
#     plot_epochs = []
#     plot_aucs = []
#     plot_auprs = []
#
#     # --- 【新增】早停变量初始化 ---
#     if use_early_stopping and args.early_stopping_patience > 0:
#         print(f"提示: 早停已启用，耐心值为 {patience} 个 epochs。")
#         best_auc = 0.0
#         epochs_no_improve = 0
#         best_model_state_dict = None
#         best_decoder_state_dict = None
#     # -----------------------------
#
#     for epoch in range(epochs):
#         if (epoch + 1) % 100 == 0:
#             start_time_20 = time.time()
#
#         model.train()
#         decoder.train()
#
#         optimizer.zero_grad()
#         embeddings, X = model(
#             (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#             edge_index, edge_weight
#         )
#         embeddings.retain_grad()
#
#         permutation = np.random.permutation(num_samples)
#         total_loss = 0.0
#
#         for i in range(0, num_samples, batch_size):
#             idx = permutation[i:i + batch_size]
#             batch_drug_idx = drug_idx[idx]
#             batch_microbe_idx = microbe_idx[idx]
#             batch_labels = labels[idx]
#             drug_emb = embeddings[batch_drug_idx]
#             microbe_emb = embeddings[microbe_offset + batch_microbe_idx]
#             logits = decoder(drug_emb, microbe_emb)
#             loss = criterion(logits, torch.tensor(batch_labels, dtype=torch.float32, device=device))
#             total_loss += loss
#
#         total_loss.backward()
#         optimizer.step()
#         # ======================== 【在这里新增】 ========================
#         # 2. 在每个epoch结束后，更新学习率
#         scheduler.step()
#         # ===============================================================
#
#         if epoch == epochs - 1:
#             last_epoch_gradients = embeddings.grad.detach().cpu().numpy() if embeddings.grad is not None else None
#             final_embeddings = embeddings.detach()
#             final_X = X.detach()
#
#         if (epoch + 1) % 6 == 0:
#             if test_data is not None:
#                 model.eval()
#                 decoder.eval()
#                 with torch.no_grad():
#                     test_auc, test_aupr = evaluate_gcn(
#                         model, decoder, test_data, edge_index, edge_weight,
#                         drug_fg, drug_features, drug_bert,
#                         microbe_features, microbe_bert, microbe_path,
#                         microbe_offset, device
#                     )
#                 plot_epochs.append(epoch + 1)
#                 plot_aucs.append(test_auc)
#                 plot_auprs.append(test_aupr)
#
#                 # --- 【将你的早停代码块粘贴在这里】 ---
#                 if use_early_stopping and patience > 0:
#                     if test_auc > best_auc:
#                         best_auc = test_auc
#                         epochs_no_improve = 0
#                         # 使用 deepcopy 保存最佳模型状态
#                         best_model_state_dict = copy.deepcopy(model.state_dict())
#                         best_decoder_state_dict = copy.deepcopy(decoder.state_dict())
#                     else:
#                         epochs_no_improve += 1
#
#                     if epochs_no_improve >= patience:
#                         print(f"早停触发: 在 {epoch + 1} 个 epochs 后，验证集 AUC 连续 {patience} 次未提升。")
#                         break  # 退出训练循环
#                 # ------------------------------------
#
#
#                 model.train()
#                 decoder.train()
#
#         if (epoch + 1) % 100 == 0:
#             end_time_20 = time.time()
#             elapsed_20 = end_time_20 - start_time_20
#             avg_loss = total_loss.item() / (num_samples / batch_size)
#             output_string = f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, 20-epoch time: {elapsed_20:.2f} sec"
#             if test_data is not None:
#                 model.eval()
#                 decoder.eval()
#                 with torch.no_grad():
#                     test_auc, test_aupr = evaluate_gcn(
#                         model, decoder, test_data, edge_index, edge_weight,
#                         drug_fg, drug_features, drug_bert,
#                         microbe_features, microbe_bert, microbe_path,
#                         microbe_offset, device
#                     )
#                 output_string += f", Test AUC: {test_auc:.4f}, Test AUPR: {test_aupr:.4f}"
#             print(output_string)
#
#         # --- 【新增】加载最佳模型 ---
#     if use_early_stopping and args.early_stopping_patience > 0 and best_model_state_dict is not None:
#         print(f"加载早停找到的最佳模型 (Test AUC: {best_auc:.4f})")
#         model.load_state_dict(best_model_state_dict)
#         decoder.load_state_dict(best_decoder_state_dict)
#     # ---------------------------
#
#
#     # <--- 【核心修改2】: 修改绘图和保存逻辑 --->
#     if test_data is not None and plot_epochs and plot_filename:
#         # 强制使用英文字体，避免中文缺失问题
#         plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
#         plt.rcParams['axes.unicode_minus'] = False
#
#         plt.figure(figsize=(12, 8))
#         plt.plot(plot_epochs, plot_aucs, marker='o', linestyle='-', label='Test AUC')
#         plt.plot(plot_epochs, plot_auprs, marker='s', linestyle='--', label='Test AUPR')
#         # 文件名已经能区分fold和阶段，标题可以更通用或也包含这些信息
#         plt.title(f'Training Curve ({os.path.splitext(plot_filename)[0]})', fontsize=16)
#         plt.xlabel('Epoch', fontsize=12)
#         plt.ylabel('Score', fontsize=12)
#         plt.legend(fontsize=12)
#         plt.grid(True)
#
#         os.makedirs(save_dir, exist_ok=True)
#         # 使用传入的文件名构造完整路径
#         save_path = os.path.join(save_dir, plot_filename)
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         print(f"成功: 训练曲线图已保存 -> {save_path}")
#     # <--- 【核心修改2 结束】 --->
#
#     return model, decoder, last_epoch_gradients, final_embeddings, final_X
#


# def train_gcn(
#         train_data, edge_index, edge_weight,
#         drug_fg, drug_features, drug_bert,
#         microbe_features, microbe_bert, microbe_path, microbe_offset,
#         epochs=100, lr=0.01, hidden=64, dropout=0.5,
#         args=None, device='cpu', batch_size=256,
#         test_data=None,
#         fold_num=0,
#         save_dir='.',
#         plot_filename=None,
#         weight_decay=0.0,
#         use_early_stopping=False,
#         patience=50
# ):
#     model = GCNWithMLP(
#         drug_in_dim=drug_fg.shape[1],
#         drug_out_dim=drug_fg.shape[0],
#         microbe_dim=microbe_features.shape[1],
#         microbe_out_dim=microbe_features.shape[1],
#         gcn_hidden=hidden,
#         dropout=dropout,
#         use_microbe_mlp=False,
#         dataset_name=args.dataset
#     ).to(device)
#
#     decoder = MLPDecoder(hidden).to(device)
#     optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=args.wd_retrain)
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer,
#         step_size=args.lr_step_size,
#         gamma=args.lr_gamma
#     )
#     criterion = nn.BCEWithLogitsLoss()
#
#     # ==================== 【核心修改：初始化混合精度组件】 ====================
#     # 仅当数据集为 aBiofilm 且在 CUDA 上运行时启用，现在加上MDAD
#     use_amp = (args.dataset in ['aBiofilm', 'MDAD']) and ('cuda' in str(device))
#
#     scaler = GradScaler(enabled=use_amp)
#     if use_amp:
#         print(f"提示: 已为 aBiofilm 数据集启用混合精度训练 (Fold {fold_num + 1})。")
#     # ========================================================================
#
#     # 特征转换只需要一次
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)
#
#     drug_idx, microbe_idx, labels = train_data
#     drug_idx = np.array(drug_idx)
#     microbe_idx = np.array(microbe_idx)
#     labels = np.array(labels)
#     num_samples = len(drug_idx)
#
#     last_epoch_gradients = None
#     final_embeddings = None
#     final_X = None
#
#     plot_epochs = []
#     plot_aucs = []
#     plot_auprs = []
#
#     if use_early_stopping and args.early_stopping_patience > 0:
#         print(f"提示: 早停已启用，耐心值为 {patience} 个 epochs。")
#         best_auc = 0.0
#         epochs_no_improve = 0
#         best_model_state_dict = None
#         best_decoder_state_dict = None
#
#     for epoch in range(epochs):
#         if (epoch + 1) % 100 == 0:
#             start_time_20 = time.time()
#
#         model.train()
#         decoder.train()
#
#         optimizer.zero_grad()
#
#         # ==================== 【核心修改：应用混合精度训练】 ====================
#         # 使用 autocast 上下文管理器包裹前向传播和损失计算
#         with autocast(enabled=use_amp):
#             embeddings, X = model(
#                 (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#                 edge_index, edge_weight
#             )
#             embeddings.retain_grad()
#
#             permutation = np.random.permutation(num_samples)
#             total_loss = 0.0
#
#             for i in range(0, num_samples, batch_size):
#                 idx = permutation[i:i + batch_size]
#                 batch_drug_idx = drug_idx[idx]
#                 batch_microbe_idx = microbe_idx[idx]
#                 batch_labels = labels[idx]
#                 drug_emb = embeddings[batch_drug_idx]
#                 microbe_emb = embeddings[microbe_offset + batch_microbe_idx]
#                 logits = decoder(drug_emb, microbe_emb)
#                 loss = criterion(logits, torch.tensor(batch_labels, dtype=torch.float32, device=device))
#                 total_loss += loss
#
#         # 使用 GradScaler 缩放损失、反向传播和更新优化器
#         scaler.scale(total_loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         # ========================================================================
#
#         scheduler.step()
#
#         if epoch == epochs - 1:
#             last_epoch_gradients = embeddings.grad.detach().cpu().numpy() if embeddings.grad is not None else None
#             final_embeddings = embeddings.detach()
#             final_X = X.detach()
#
#         if (epoch + 1) % 6 == 0:
#             if test_data is not None:
#                 model.eval()
#                 decoder.eval()
#                 with torch.no_grad():
#                     test_auc, test_aupr = evaluate_gcn(
#                         model, decoder, test_data, edge_index, edge_weight,
#                         drug_fg, drug_features, drug_bert,
#                         microbe_features, microbe_bert, microbe_path,
#                         microbe_offset, device
#                     )
#                 plot_epochs.append(epoch + 1)
#                 plot_aucs.append(test_auc)
#                 plot_auprs.append(test_aupr)
#
#                 if use_early_stopping and patience > 0:
#                     if test_auc > best_auc:
#                         best_auc = test_auc
#                         epochs_no_improve = 0
#                         best_model_state_dict = copy.deepcopy(model.state_dict())
#                         best_decoder_state_dict = copy.deepcopy(decoder.state_dict())
#                     else:
#                         epochs_no_improve += 1
#
#                     if epochs_no_improve >= patience:
#                         print(f"早停触发: 在 {epoch + 1} 个 epochs 后，验证集 AUC 连续 {patience} 次未提升。")
#                         break
#                 model.train()
#                 decoder.train()
#
#         if (epoch + 1) % 100 == 0:
#             end_time_20 = time.time()
#             elapsed_20 = end_time_20 - start_time_20
#             avg_loss = total_loss.item() / (num_samples / batch_size)
#             output_string = f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, 20-epoch time: {elapsed_20:.2f} sec"
#             if test_data is not None:
#                 model.eval()
#                 decoder.eval()
#                 with torch.no_grad():
#                     test_auc, test_aupr = evaluate_gcn(
#                         model, decoder, test_data, edge_index, edge_weight,
#                         drug_fg, drug_features, drug_bert,
#                         microbe_features, microbe_bert, microbe_path,
#                         microbe_offset, device
#                     )
#                 output_string += f", Test AUC: {test_auc:.4f}, Test AUPR: {test_aupr:.4f}"
#             print(output_string)
#
#     if use_early_stopping and args.early_stopping_patience > 0 and best_model_state_dict is not None:
#         print(f"加载早停找到的最佳模型 (Test AUC: {best_auc:.4f})")
#         model.load_state_dict(best_model_state_dict)
#         decoder.load_state_dict(best_decoder_state_dict)
#
#     if test_data is not None and plot_epochs and plot_filename:
#         plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
#         plt.rcParams['axes.unicode_minus'] = False
#         plt.figure(figsize=(12, 8))
#         plt.plot(plot_epochs, plot_aucs, marker='o', linestyle='-', label='Test AUC')
#         plt.plot(plot_epochs, plot_auprs, marker='s', linestyle='--', label='Test AUPR')
#         plt.title(f'Training Curve ({os.path.splitext(plot_filename)[0]})', fontsize=16)
#         plt.xlabel('Epoch', fontsize=12)
#         plt.ylabel('Score', fontsize=12)
#         plt.legend(fontsize=12)
#         plt.grid(True)
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, plot_filename)
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         print(f"成功: 训练曲线图已保存 -> {save_path}")
#
#     return model, decoder, last_epoch_gradients, final_embeddings, final_X
#
#


# train_eval.py

# ... (文件顶部的 import 保持不变) ...

# 请用下面的函数【整体替换】你文件中旧的 train_gcn 函数

def train_gcn(
        train_data, edge_index, edge_weight,
        drug_fg, drug_features, drug_bert,
        microbe_features, microbe_bert, microbe_path, microbe_offset,
        epochs=100, lr=0.01, hidden=64, dropout=0.5,
        args=None, device='cpu', batch_size=256,
        test_data=None,
        fold_num=0,
        save_dir='.',
        plot_filename=None,
        weight_decay=0.0,
        use_early_stopping=False,
        patience=50
):
    torch.autograd.set_detect_anomaly(True)  # <--- 在这里添加
    model = GCNWithMLP(
        drug_in_dim=drug_fg.shape[1],
        drug_out_dim=drug_fg.shape[0],
        microbe_dim=microbe_features.shape[1],
        microbe_out_dim=microbe_features.shape[1],
        gcn_hidden=hidden,
        dropout=dropout,
        use_microbe_mlp=False,
        dataset_name=args.dataset
    ).to(device)

    decoder = MLPDecoder(hidden).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=args.wd_retrain)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    criterion = nn.BCEWithLogitsLoss()

   # use_amp = (args.dataset in ['a']) and ('cuda' in str(device))
    use_amp = True  # 彻底关闭混合精度
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print(f"提示: 已为 {args.dataset} 数据集启用混合精度训练 (Fold {fold_num + 1})。")
    else:
        print('不启用混合精度')

    drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
    drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
    drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
    microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
    microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
    microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)

    drug_idx, microbe_idx, labels = train_data
    drug_idx = np.array(drug_idx)
    microbe_idx = np.array(microbe_idx)
    labels = np.array(labels)
    num_samples = len(drug_idx)

    last_epoch_gradients = None
    final_embeddings = None
    final_X = None

    plot_epochs = []
    plot_aucs = []
    plot_auprs = []
    plot_accs = [] # 新增

    if use_early_stopping and args.early_stopping_patience > 0:
        print(f"提示: 早停已启用，耐心值为 {patience} 个 epochs。")
        best_auc = 0.0
        epochs_no_improve = 0
        best_model_state_dict = None
        best_decoder_state_dict = None

    for epoch in range(epochs):
        if (epoch + 1) % 100 == 0:
            start_time_20 = time.time()

        model.train()
        decoder.train()
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            embeddings, X = model(
                (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
                edge_index, edge_weight
            )
            embeddings.retain_grad()
            permutation = np.random.permutation(num_samples)
            total_loss = 0.0
            for i in range(0, num_samples, batch_size):
                idx = permutation[i:i + batch_size]
                batch_drug_idx = drug_idx[idx]
                batch_microbe_idx = microbe_idx[idx]
                batch_labels = labels[idx]
                drug_emb = embeddings[batch_drug_idx]
                microbe_emb = embeddings[microbe_offset + batch_microbe_idx]
                logits = decoder(drug_emb, microbe_emb)
                loss = criterion(logits, torch.tensor(batch_labels, dtype=torch.float32, device=device))
                total_loss += loss

        scaler.scale(total_loss).backward()

        # --- 添加梯度裁剪 ---
        # 在 optimizer.step() 之前 unscale 梯度
        scaler.unscale_(optimizer)
        # 对 unscale 后的梯度进行裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # --------------------


        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if epoch == epochs - 1:
            last_epoch_gradients = embeddings.grad.detach().cpu().numpy() if embeddings.grad is not None else None
            final_embeddings = embeddings.detach()
            final_X = X.detach()

        if (epoch + 1) % 6 == 0:
            if test_data is not None:
                model.eval()
                decoder.eval()
                with torch.no_grad():
                    # --- 【修改点 1】 ---
                    test_auc, test_aupr, test_acc = evaluate_gcn(
                        model, decoder, test_data, edge_index, edge_weight,
                        drug_fg, drug_features, drug_bert,
                        microbe_features, microbe_bert, microbe_path,
                        microbe_offset, device
                    )
                plot_epochs.append(epoch + 1)
                plot_aucs.append(test_auc)
                plot_auprs.append(test_aupr)
                plot_accs.append(test_acc) # 新增

                if use_early_stopping and patience > 0:
                    if test_auc > best_auc:
                        best_auc = test_auc
                        epochs_no_improve = 0
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                        best_decoder_state_dict = copy.deepcopy(decoder.state_dict())
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        print(f"早停触发: 在 {epoch + 1} 个 epochs 后，验证集 AUC 连续 {patience} 次未提升。")
                        break
                model.train()
                decoder.train()

        if (epoch + 1) % 100 == 0:
            end_time_20 = time.time()
            elapsed_20 = end_time_20 - start_time_20
            avg_loss = total_loss.item() / (num_samples / batch_size)
            output_string = f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, 20-epoch time: {elapsed_20:.2f} sec"
            if test_data is not None:
                model.eval()
                decoder.eval()
                with torch.no_grad():
                    # --- 【修改点 2】 ---
                    test_auc, test_aupr, test_acc = evaluate_gcn(
                        model, decoder, test_data, edge_index, edge_weight,
                        drug_fg, drug_features, drug_bert,
                        microbe_features, microbe_bert, microbe_path,
                        microbe_offset, device
                    )
                # --- 【修改点 3】 ---
                output_string += f", Test AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, ACC: {test_acc:.4f}"
            print(output_string)

    if use_early_stopping and args.early_stopping_patience > 0 and best_model_state_dict is not None:
        print(f"加载早停找到的最佳模型 (Test AUC: {best_auc:.4f})")
        model.load_state_dict(best_model_state_dict)
        decoder.load_state_dict(best_decoder_state_dict)

    if test_data is not None and plot_epochs and plot_filename:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 8))
        plt.plot(plot_epochs, plot_aucs, marker='o', linestyle='-', label='Test AUC')
        plt.plot(plot_epochs, plot_auprs, marker='s', linestyle='--', label='Test AUPR')
        plt.plot(plot_epochs, plot_accs, marker='^', linestyle=':', label='Test ACC') # 新增
        plt.title(f'Training Curve ({os.path.splitext(plot_filename)[0]})', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, plot_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"成功: 训练曲线图已保存 -> {save_path}")

    return model, decoder, last_epoch_gradients, final_embeddings, final_X


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

# 假设 GCNWithMLP 和 MLPDecoder 类已经定义好

# def train_gcn(
#         train_data, edge_index, edge_weight,
#         drug_fg, drug_features, drug_bert,
#         microbe_features, microbe_bert, microbe_path, microbe_offset,
#         epochs=100, lr=0.01, hidden=64, dropout=0.5,
#         args=None, device='cpu', batch_size=256
# ):
#     """
#     使用随机梯度下降（Mini-Batch SGD）训练GCN模型。
#     每个batch都会执行一次完整的前向、反向和更新步骤。
#     """
#     model = GCNWithMLP(
#         drug_in_dim=drug_fg.shape[1],
#         drug_out_dim=drug_fg.shape[0],
#         microbe_dim=microbe_features.shape[1],
#         microbe_out_dim=microbe_features.shape[1],
#         gcn_hidden=hidden,
#         dropout=dropout,
#         use_microbe_mlp=False,
#         dataset_name=args.dataset
#     ).to(device)
#
#     decoder = MLPDecoder(hidden).to(device)
#     optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)
#     criterion = nn.BCEWithLogitsLoss()
#
#     # 特征转换只需要一次
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)
#
#     drug_idx, microbe_idx, labels = train_data
#     drug_idx = np.array(drug_idx)
#     microbe_idx = np.array(microbe_idx)
#     labels = np.array(labels)
#     num_samples = len(drug_idx)
#
#     for epoch in range(epochs):
#         if epoch % 40 == 0:
#             start_time_40 = time.time()
#
#         model.train()
#         decoder.train()
#
#         permutation = np.random.permutation(num_samples)
#         epoch_loss = 0.0  # 用于记录和打印当前epoch的总损失
#
#         for i in range(0, num_samples, batch_size):
#             # 1. 清空上一轮的梯度
#             optimizer.zero_grad()
#
#             # 2. 对整个图进行前向传播，得到所有节点的嵌入
#             #    注意：这一步在每个batch都会执行，计算成本较高
#             embeddings, X = model(
#                 (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#                 edge_index, edge_weight
#             )
#
#             # 3. 获取当前batch的数据
#             idx = permutation[i:i + batch_size]
#             batch_drug_idx = drug_idx[idx]
#             batch_microbe_idx = microbe_idx[idx]
#             batch_labels = labels[idx]
#
#             # 4. 从全图嵌入中抽取当前batch所需的节点嵌入
#             drug_emb = embeddings[batch_drug_idx]
#             microbe_emb = embeddings[microbe_offset + batch_microbe_idx]
#
#             # 5. 通过解码器得到预测结果并计算损失
#             logits = decoder(drug_emb, microbe_emb)
#             loss = criterion(logits, torch.tensor(batch_labels, dtype=torch.float32, device=device))
#
#             # 6. 反向传播，计算当前batch的梯度
#             loss.backward()
#
#             # 7. 根据梯度更新模型参数
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#         if (epoch + 1) % 40 == 0:
#             end_time_40 = time.time()
#             elapsed_40 = end_time_40 - start_time_40
#             num_batches = num_samples // batch_size + int(num_samples % batch_size != 0)
#             avg_loss = epoch_loss / num_batches
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, 40-epoch time: {elapsed_40:.2f} sec")
#
#     # 训练结束后，在评估模式下计算最终的嵌入
#     model.eval()
#     decoder.eval()
#     with torch.no_grad():
#         final_embeddings, final_X = model(
#             (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#             edge_index, edge_weight
#         )
#
#     # 在SGD模式下，每个batch的梯度用完即弃，因此无法在epoch结束时获得有意义的全图梯度
#     last_epoch_gradients = None
#
#     return model, decoder, last_epoch_gradients, final_embeddings, final_X
#
# train_eval.py

# 在文件顶部，与其他 from sklearn.metrics... 一起添加 accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

# def evaluate_gcn(model, decoder, test_data, edge_index, edge_weight, drug_fg,drug_features,drug_bert, microbe_features,microbe_bert,microbe_path,microbe_offset, device='cpu',return_probs=False):
#     model.eval()
#     decoder.eval()
#     drug_idx, microbe_idx, labels = test_data
#     #drug_fg,drug_features,drug_bert= torch.tensor(drug_fg,drug_features,drug_bert,dtype=torch.float32).to(device)
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#
#     #microbe_feat = torch.tensor(microbe_features, dtype=torch.float32).to(device)
#
#
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32).to(device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32).to(device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32).to(device)
#     with torch.no_grad():
#         with torch.no_grad():
#             #adj = torch.tensor(A, dtype=torch.float32).to(device)
#
#             #embeddings ,X= model((drug_fg,drug_features,drug_bert,microbe_features,microbe_bert,microbe_path), adj)  # 直接forward
#             embeddings, X = model((drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#                                   edge_index, edge_weight)
#
#             drug_emb = embeddings[drug_idx]
#             microbe_emb = embeddings[microbe_idx + microbe_offset]
#             logits = decoder(drug_emb, microbe_emb)
#             probs = torch.sigmoid(logits).cpu().numpy()
#
#     auc = roc_auc_score(labels, probs)
#     aupr = average_precision_score(labels, probs)
#     # 【修改2】根据新参数决定返回值
#     if return_probs:
#         return auc, aupr, probs
#     else:
#         return auc, aupr
#
# train_eval.py

# ... (文件其他部分保持不变) ...

# 请用这个函数【整体替换】旧的 evaluate_gcn 函数
def evaluate_gcn(model, decoder, data, edge_index, edge_weight,
                 drug_fg, drug_features, drug_bert,
                 microbe_features, microbe_bert, microbe_path,
                 microbe_offset, device,
                 return_probs=False):  # <--- 核心修改1：增加新参数并设置默认值
    """
    评估GCN模型性能。
    新增功能：如果 return_probs=True，则除了返回指标外，还返回预测概率。
    """
    model.eval()
    decoder.eval()

    drug_idx, microbe_idx, labels = data

    with torch.no_grad():
        # 确保所有输入特征都是Tensor
        # 如果已经是Tensor，再次调用torch.tensor会创建一个副本，但类型是正确的
        # 如果是Numpy，则会进行转换
        features = (
            torch.as_tensor(drug_fg, dtype=torch.float32, device=device),
            torch.as_tensor(drug_features, dtype=torch.float32, device=device),
            torch.as_tensor(drug_bert, dtype=torch.float32, device=device),
            torch.as_tensor(microbe_features, dtype=torch.float32, device=device),
            torch.as_tensor(microbe_bert, dtype=torch.float32, device=device),
            torch.as_tensor(microbe_path, dtype=torch.float32, device=device),
        )

        embeddings, _ = model(features, edge_index, edge_weight)

        x_drug = embeddings[drug_idx]
        x_microbe = embeddings[microbe_idx + microbe_offset]

        # 确保 x_drug 和 x_microbe 在正确的设备上
        x_drug = x_drug.to(device)
        x_microbe = x_microbe.to(device)

        preds = decoder(x_drug, x_microbe).squeeze()

        # 将预测和标签移到CPU上进行指标计算
        all_probs = torch.sigmoid(preds).cpu().numpy()
        all_labels = labels

    auc = roc_auc_score(all_labels, all_probs)
    aupr = average_precision_score(all_labels, all_probs)

    # 计算ACC
    predicted_classes = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, predicted_classes)

    # <--- 核心修改2：根据新参数决定返回什么
    if return_probs:
        # 如果调用者需要概率值，则返回 AUC, AUPR 和概率数组
        # 注意：这里返回了3个值，与 main.py 中的接收变量数量 (train_auc, train_aupr, train_probs) 对应
        return auc, aupr, all_probs
    else:
        # 否则，保持原来的行为，返回 AUC, AUPR 和 ACC
        return auc, aupr, acc


#定义一个计算Fisher信息的函数。Fisher信息矩阵反映了每个模型参数对损失函数的敏感性，我们将通过计算每个参数的二阶梯度来得到这个矩阵。
def compute_fisher(model, decoder, data, A, drug_features, microbe_features, microbe_offset, device):
    model.eval()
    decoder.eval()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    fisher_decoder = {name: torch.zeros_like(p) for name, p in decoder.named_parameters()}

    # 用一部分数据估算
    drug_idx, microbe_idx, labels = data
    drug_feat = torch.tensor(drug_features, dtype=torch.float32).to(device)
    microbe_feat = torch.tensor(microbe_features, dtype=torch.float32).to(device)
    drug_feat_reduced = model.mlp(drug_feat)
    X = build_gcn_features(drug_feat_reduced.detach().cpu().numpy(), microbe_feat.detach().cpu().numpy())
    X = torch.tensor(X, dtype=torch.float32).to(device)
    adj = torch.tensor(A, dtype=torch.float32).to(device)
    embeddings = model.gcn(X, adj)
    drug_emb = embeddings[drug_idx]
    microbe_emb = embeddings[microbe_offset + microbe_idx]
    logits = decoder(drug_emb, microbe_emb)
    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor(labels, dtype=torch.float32).to(device))

    # 计算每个参数的梯度平方（近似Fisher）
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            fisher[name] += (p.grad.detach() ** 2)
    for name, p in decoder.named_parameters():
        if p.grad is not None:
            fisher_decoder[name] += (p.grad.detach() ** 2)
    return fisher, fisher_decoder

#新增 EWC 损失函数
# def ewc_loss_fn(model, decoder, old_params, old_params_decoder, fisher, fisher_decoder, lambda_ewc):
#     ewc_loss = 0
#     for name, param in model.named_parameters():
#         if name in old_params and name in fisher:
#             if param.shape == old_params[name].shape and param.shape == fisher[name].shape:
#                 ewc_loss += (fisher[name] * (param - old_params[name]) ** 2).sum()
#             #else:
#                 print(f"[EWC][SKIP] name: {name} shape: {param.shape}, old: {old_params[name].shape}, fisher: {fisher[name].shape}")
#         else:
#             print(f"[EWC][SKIP] name: {name} not found in old_params/fisher")
#     # decoder同理
#     for name, param in decoder.named_parameters():
#         if name in old_params_decoder and name in fisher_decoder:
#             if param.shape == old_params_decoder[name].shape and param.shape == fisher_decoder[name].shape:
#                 ewc_loss += (fisher_decoder[name] * (param - old_params_decoder[name]) ** 2).sum()
#             #else:
#                 print(f"[EWC][DECODER][SKIP] name: {name} shape: {param.shape}, old: {old_params_decoder[name].shape}, fisher: {fisher_decoder[name].shape}")
#         #else:
#             print(f"[EWC][DECODER][SKIP] name: {name} not found in old_params_decoder/fisher_decoder")
#     return lambda_ewc * ewc_loss
def ewc_loss_fn(model, decoder, old_params, old_params_decoder, fisher, fisher_decoder, lambda_ewc, print_once_set=None):
    ewc_loss = 0.0
    if print_once_set is None:
        print_once_set = set()
    # 只对gcn.相关参数
    for name, param in model.named_parameters():
        if 'gcn.conv2' in name:
            if name in old_params and name in fisher:
                if param.shape == old_params[name].shape and param.shape == fisher[name].shape:
                    ewc_loss += (fisher[name] * (param - old_params[name]) ** 2).sum()
                    if name not in print_once_set:
                        #print(f"[EWC][APPLY] model: {name} shape: {param.shape}")
                        print_once_set.add(name)
    for name, param in decoder.named_parameters():
        if name in old_params_decoder and name in fisher_decoder:
            if param.shape == old_params_decoder[name].shape and param.shape == fisher_decoder[name].shape:
                ewc_loss += (fisher_decoder[name] * (param - old_params_decoder[name]) ** 2).sum()
                if name not in print_once_set:
                    #print(f"[EWC][APPLY] decoder: {name} shape: {param.shape}")
                    print_once_set.add(name)
    return lambda_ewc * ewc_loss





def compute_fisher_gcn(model, decoder, data, edge_index, edge_weight, drug_fg, drug_features, drug_bert,
                       microbe_features, microbe_bert, microbe_path, microbe_offset, device):
    """
    计算Fisher信息矩阵（与GCN兼容的版本）
    """
    model.eval()
    decoder.eval()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    fisher_decoder = {name: torch.zeros_like(p) for name, p in decoder.named_parameters() if p.requires_grad}

    drug_idx, microbe_idx, labels = data

    # 转换为tensor
    drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
    drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
    drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
    microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
    microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
    microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)

    # 前向传播
    embeddings, X = model((drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
                          edge_index, edge_weight)
    drug_emb = embeddings[drug_idx]
    microbe_emb = embeddings[microbe_offset + microbe_idx]
    logits = decoder(drug_emb, microbe_emb)
    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor(labels, dtype=torch.float32, device=device))

    # 计算梯度
    model.zero_grad()
    decoder.zero_grad()
    loss.backward()

    num_samples = len(drug_idx)  # ★ 加这一行
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            fisher[name] += (p.grad.detach() ** 2) / num_samples  # ★ 这里除以样本数
    for name, p in decoder.named_parameters():
        if p.requires_grad and p.grad is not None:
            fisher_decoder[name] += (p.grad.detach() ** 2) / num_samples  # ★ 同上
    # # 保存梯度平方作为Fisher信息近似
    # for name, p in model.named_parameters():
    #     if p.requires_grad and p.grad is not None:
    #         fisher[name] += (p.grad.detach() ** 2)
    # for name, p in decoder.named_parameters():
    #     if p.requires_grad and p.grad is not None:
    #         fisher_decoder[name] += (p.grad.detach() ** 2)



    return fisher, fisher_decoder



#
# def train_gcn_ewc_new(
#         train_data, edge_index, edge_weight, drug_fg, drug_features, drug_bert,
#         microbe_features, microbe_bert, microbe_path, microbe_offset,
#         epochs, lr, hidden, dropout, device,
#         old_params, old_params_decoder, fisher, fisher_decoder, lambda_ewc,
#         model=None, decoder=None, args=None
# ):
#     """
#     支持EWC的GCN训练函数（新版本，与主程序兼容）
#     """
#     model = model.to(device)
#     decoder = decoder.to(device)
#
#     # 只优化需要训练的参数
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, list(model.parameters()) + list(decoder.parameters())),
#         lr=lr
#     )
#     criterion = nn.BCEWithLogitsLoss()
#     model.train()
#     decoder.train()
#
#     # 转换为tensor（一次性转换）
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)
#
#     print_once_set = set()  # 用于控制EWC日志只打印一次
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         drug_idx, microbe_idx, labels = train_data
#
#         # 获取GCN嵌入
#         embeddings, X = model((drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#                               edge_index, edge_weight)
#
#         drug_emb = embeddings[drug_idx]
#         microbe_emb = embeddings[microbe_offset + microbe_idx]
#         logits = decoder(drug_emb, microbe_emb)
#
#         # 主任务损失
#         main_loss = criterion(logits, torch.tensor(labels, dtype=torch.float32, device=device))
#
#         # EWC损失
#         ewc_loss = ewc_loss_fn(model, decoder, old_params, old_params_decoder,
#                                fisher, fisher_decoder, lambda_ewc, print_once_set)
#
#         total_loss = main_loss + ewc_loss
#         total_loss.backward()
#         optimizer.step()
#
#         if (epoch + 1) % 40 == 0:
#             print(
#                 f"[EWC] Epoch {epoch + 1}/{epochs}, Main Loss: {main_loss.item():.4f}, EWC Loss: {ewc_loss.item():.4f}, Total: {total_loss.item():.4f}")
#
#     return model, decoder
# =================================================================================
#  请在 train_eval.py 文件中，用以下完整代码替换旧的 train_gcn_ewc_new 函数
# =================================================================================

# def train_gcn_ewc_new(
#         # --- DrugVirus (新任务) 数据 ---
#         train_data, edge_index, edge_weight, drug_fg, drug_features, drug_bert,
#         microbe_features, microbe_bert, microbe_path, microbe_offset,
#
#         # --- MDAD (旧任务) 数据 (新增参数) ---
#         mdad_train_data, mdad_edge_index, mdad_edge_weight,
#         mdad_drug_fg_norm, mdad_drug_features_norm, mdad_drug_bert_norm,
#         mdad_microbe_features_norm, mdad_microbe_bert_norm, mdad_microbe_path_norm,
#         mdad_microbe_offset,
#
#         # --- 训练超参数 ---
#         epochs, lr, hidden, dropout, device,
#
#         # --- EWC 相关参数 ---
#         old_params, old_params_decoder, fisher, fisher_decoder, lambda_ewc,
#
#         # --- 其他 ---
#         model=None, decoder=None, args=None,
#
#         # --- 新增：任务权重超参数 ---
#         alpha=0.5,
#     # --- 新增：权重衰减 ---
#         weight_decay = 1e-5  # <--- 【核心修改1：增加参数】
# ):
#     """
#     支持EWC和多任务排练的GCN训练函数。
#     """
#     model = model.to(device)
#     decoder = decoder.to(device)
#
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, list(model.parameters()) + list(decoder.parameters())),
#         lr=lr,
#         weight_decay=weight_decay
#     )
#     # ======================== 【在这里新增】 ========================
#     # 1. 创建学习率调度器
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer,
#         step_size=args.lr_step_size,
#         gamma=args.lr_gamma
#     )
#     # ===============================================================
#
#     criterion = nn.BCEWithLogitsLoss()
#     model.train()
#     decoder.train()
#
#     # --- 一次性转换 DrugVirus 特征为 Tensor ---
#     drug_fg = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#     microbe_features = torch.tensor(microbe_features, dtype=torch.float32, device=device)
#     microbe_bert = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
#     microbe_path = torch.tensor(microbe_path, dtype=torch.float32, device=device)
#
#     # --- 一次性转换 MDAD 特征为 Tensor (新增) ---
#     mdad_drug_fg_norm = torch.tensor(mdad_drug_fg_norm, dtype=torch.float32, device=device)
#     mdad_drug_features_norm = torch.tensor(mdad_drug_features_norm, dtype=torch.float32, device=device)
#     mdad_drug_bert_norm = torch.tensor(mdad_drug_bert_norm, dtype=torch.float32, device=device)
#     mdad_microbe_features_norm = torch.tensor(mdad_microbe_features_norm, dtype=torch.float32, device=device)
#     mdad_microbe_bert_norm = torch.tensor(mdad_microbe_bert_norm, dtype=torch.float32, device=device)
#     mdad_microbe_path_norm = torch.tensor(mdad_microbe_path_norm, dtype=torch.float32, device=device)
#
#     print_once_set = set()
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#
#         # ==================== 核心修改区域开始 ====================
#
#         # --- 1. DrugVirus (新任务) 前向传播和损失计算 ---
#         drug_idx, microbe_idx, labels = train_data
#         embeddings_dv, _ = model(
#             (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path),
#             edge_index, edge_weight
#         )
#         drug_emb_dv = embeddings_dv[drug_idx]
#         microbe_emb_dv = embeddings_dv[microbe_offset + microbe_idx]
#         logits_dv = decoder(drug_emb_dv, microbe_emb_dv)
#         loss_drugvirus = criterion(logits_dv, torch.tensor(labels, dtype=torch.float32, device=device))
#
#         # --- 2. MDAD (旧任务) 前向传播和损失计算 (新增) ---
#         mdad_drug_idx, mdad_microbe_idx, mdad_labels = mdad_train_data
#         embeddings_mdad, _ = model(
#             (mdad_drug_fg_norm, mdad_drug_features_norm, mdad_drug_bert_norm,
#              mdad_microbe_features_norm, mdad_microbe_bert_norm, mdad_microbe_path_norm),
#             mdad_edge_index, mdad_edge_weight
#         )
#         drug_emb_mdad = embeddings_mdad[mdad_drug_idx]
#         microbe_emb_mdad = embeddings_mdad[mdad_microbe_offset + mdad_microbe_idx]
#         logits_mdad = decoder(drug_emb_mdad, microbe_emb_mdad)
#         loss_mdad = criterion(logits_mdad, torch.tensor(mdad_labels, dtype=torch.float32, device=device))
#
#         # --- 3. EWC 损失计算 (不变) ---
#         ewc_loss = ewc_loss_fn(model, decoder, old_params, old_params_decoder,
#                                fisher, fisher_decoder, lambda_ewc, print_once_set)
#
#         # --- 4. 构建最终的总损失 (修改) ---
#         # 总损失 = (1-alpha)*新任务损失 + alpha*旧任务损失 + EWC惩罚
#         total_loss = (1 - alpha) * loss_drugvirus + alpha * loss_mdad + ewc_loss
#
#         # ==================== 核心修改区域结束 ====================
#
#         total_loss.backward()
#         optimizer.step()
#         # ======================== 【在这里新增】 ========================
#         # 2. 在每个epoch结束后，更新学习率
#         scheduler.step()
#         # ===============================================================
#
#         if (epoch + 1) % 40 == 0:
#             print(
#                 f"[EWC-MTL] Epoch {epoch + 1}/{epochs}, "
#                 f"Loss_DV: {loss_drugvirus.item():.4f}, "
#                 f"Loss_MDAD: {loss_mdad.item():.4f}, "
#                 f"Loss_EWC: {ewc_loss.item():.4f}, "
#                 f"Total: {total_loss.item():.4f}"
#             )
#
#     return model, decoder

# 替换 train_eval.py 中旧的 train_gcn_ewc_new 函数
# def train_gcn_ewc_new(
#         # --- DrugVirus (新任务) 数据 ---
#         train_data, edge_index, edge_weight, drug_fg, drug_features, drug_bert,
#         microbe_features, microbe_bert, microbe_path, microbe_offset,
#
#         # --- MDAD (旧任务) 数据 (新增参数) ---
#         mdad_test_data,mdad_train_data, mdad_edge_index, mdad_edge_weight,
#         mdad_drug_fg_norm, mdad_drug_features_norm, mdad_drug_bert_norm,
#         mdad_microbe_features_norm, mdad_microbe_bert_norm, mdad_microbe_path_norm,
#         mdad_microbe_offset,
#
#
#
#
#
#         # --- 训练超参数 ---
#         epochs, lr, hidden, dropout, device,
#
#         # --- EWC 相关参数 ---
#         old_params, old_params_decoder, fisher, fisher_decoder, lambda_ewc,
#
#         # --- 其他 ---
#         model=None, decoder=None, args=None,
#
#
#
#         # --- 新增：任务权重超参数 ---
#         alpha=0.5,
#         weight_decay=1e-5,
#
#         # ==================== 【核心修改1：增加绘图相关参数】 ====================
#         drugvirus_test_data=None,
#         fold_num=0,
#         save_dir='.'
#         # ====================================================================
# ):
#     """
#     支持EWC和多任务排练的GCN训练函数。
#     """
#     model = model.to(device)
#     decoder = decoder.to(device)
#
#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, list(model.parameters()) + list(decoder.parameters())),
#         lr=lr,
#         weight_decay=weight_decay
#     )
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer,
#         step_size=args.lr_step_size,
#         gamma=args.lr_gamma
#     )
#
#     criterion = nn.BCEWithLogitsLoss()
#     model.train()
#     decoder.train()
#
#     # --- 一次性转换 DrugVirus 特征为 Tensor ---
#     drug_fg_t = torch.tensor(drug_fg, dtype=torch.float32, device=device)
#     drug_features_t = torch.tensor(drug_features, dtype=torch.float32, device=device)
#     drug_bert_t = torch.tensor(drug_bert, dtype=torch.float32, device=device)
#     microbe_features_t = torch.tensor(microbe_features, dtype=torch.float32, device=device)
#     microbe_bert_t = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
#     microbe_path_t = torch.tensor(microbe_path, dtype=torch.float32, device=device)
#
#     # --- 一次性转换 MDAD 特征为 Tensor ---
#     mdad_drug_fg_norm_t = torch.tensor(mdad_drug_fg_norm, dtype=torch.float32, device=device)
#     mdad_drug_features_norm_t = torch.tensor(mdad_drug_features_norm, dtype=torch.float32, device=device)
#     mdad_drug_bert_norm_t = torch.tensor(mdad_drug_bert_norm, dtype=torch.float32, device=device)
#     mdad_microbe_features_norm_t = torch.tensor(mdad_microbe_features_norm, dtype=torch.float32, device=device)
#     mdad_microbe_bert_norm_t = torch.tensor(mdad_microbe_bert_norm, dtype=torch.float32, device=device)
#     mdad_microbe_path_norm_t = torch.tensor(mdad_microbe_path_norm, dtype=torch.float32, device=device)
#
#     print_once_set = set()
#
#     # ==================== 【核心修改2：初始化绘图列表】 ====================
#     plot_epochs = []
#     plot_aucs = []
#     plot_auprs = []
#     # ====================================================================
#     mdad_plot_epochs = []
#     mdad_plot_aucs = []
#     mdad_plot_auprs = []
#
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#
#         # --- 1. DrugVirus (新任务) 前向传播和损失计算 ---
#         drug_idx, microbe_idx, labels = train_data
#         embeddings_dv, _ = model(
#             (drug_fg_t, drug_features_t, drug_bert_t, microbe_features_t, microbe_bert_t, microbe_path_t),
#             edge_index, edge_weight
#         )
#         drug_emb_dv = embeddings_dv[drug_idx]
#         microbe_emb_dv = embeddings_dv[microbe_offset + microbe_idx]
#         logits_dv = decoder(drug_emb_dv, microbe_emb_dv)
#         loss_drugvirus = criterion(logits_dv, torch.tensor(labels, dtype=torch.float32, device=device))
#
#         # --- 2. MDAD (旧任务) 前向传播和损失计算 ---
#         mdad_drug_idx, mdad_microbe_idx, mdad_labels = mdad_train_data
#         embeddings_mdad, _ = model(
#             (mdad_drug_fg_norm_t, mdad_drug_features_norm_t, mdad_drug_bert_norm_t,
#              mdad_microbe_features_norm_t, mdad_microbe_bert_norm_t, mdad_microbe_path_norm_t),
#             mdad_edge_index, mdad_edge_weight
#         )
#         drug_emb_mdad = embeddings_mdad[mdad_drug_idx]
#         microbe_emb_mdad = embeddings_mdad[mdad_microbe_offset + mdad_microbe_idx]
#         logits_mdad = decoder(drug_emb_mdad, microbe_emb_mdad)
#         loss_mdad = criterion(logits_mdad, torch.tensor(mdad_labels, dtype=torch.float32, device=device))
#
#         # --- 3. EWC 损失计算 ---
#         ewc_loss = ewc_loss_fn(model, decoder, old_params, old_params_decoder,
#                                fisher, fisher_decoder, lambda_ewc, print_once_set)
#
#         # --- 4. 构建最终的总损失 ---
#         total_loss = (1 - alpha) * loss_drugvirus + alpha * loss_mdad + ewc_loss
#
#         total_loss.backward()
#         optimizer.step()
#         scheduler.step()
#
#         # ==================== 【核心修改3：定期评估并记录数据】 ====================
#         if (epoch + 1) % 6 == 0 and drugvirus_test_data is not None:
#             model.eval()
#             decoder.eval()
#             with torch.no_grad():
#                 # 注意：这里评估的是 DrugVirus 的性能
#                 test_auc, test_aupr = evaluate_gcn(
#                     model, decoder, drugvirus_test_data, edge_index, edge_weight,
#                     drug_fg, drug_features, drug_bert,
#                     microbe_features, microbe_bert, microbe_path,
#                     microbe_offset, device
#                 )
#             plot_epochs.append(epoch + 1)
#             plot_aucs.append(test_auc)
#             plot_auprs.append(test_aupr)
#
#             mdad_auc, mdad_aupr = evaluate_gcn(
#                 model, decoder, mdad_test_data, mdad_edge_index, mdad_edge_weight,
#                 mdad_drug_fg_norm, mdad_drug_features_norm, mdad_drug_bert_norm,
#                 mdad_microbe_features_norm, mdad_microbe_bert_norm, mdad_microbe_path_norm,
#                 mdad_microbe_offset, device
#             )
#             mdad_plot_epochs.append(epoch + 1)
#             mdad_plot_aucs.append(mdad_auc)
#             mdad_plot_auprs.append(mdad_aupr)
#
#
#             model.train()  # 切换回训练模式
#             decoder.train()
#         # ========================================================================
#
#         if (epoch + 1) % 40 == 0:
#             print(
#                 f"[EWC-MTL Fold {fold_num + 1}] Epoch {epoch + 1}/{epochs}, "
#                 f"Loss_DV: {loss_drugvirus.item():.4f}, "
#                 f"Loss_MDAD: {loss_mdad.item():.4f}, "
#                 f"Loss_EWC: {ewc_loss.item():.4f}, "
#                 f"Total: {total_loss.item():.4f}"
#             )
#
#
#     # ==================== 【核心修改4：训练结束后绘图并保存】 ====================
#     if plot_epochs:
#         plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
#         plt.rcParams['axes.unicode_minus'] = False
#
#         plt.figure(figsize=(12, 8))
#         plt.plot(plot_epochs, plot_aucs, marker='o', linestyle='-', label='DrugVirus Test AUC')
#         plt.plot(plot_epochs, plot_auprs, marker='s', linestyle='--', label='DrugVirus Test AUPR')
#         plt.title(f'Incremental Learning Curve (Fold {fold_num + 1})', fontsize=16)
#         plt.xlabel('Epoch', fontsize=12)
#         plt.ylabel('Score', fontsize=12)
#         plt.legend(fontsize=12)
#         plt.grid(True)
#
#         os.makedirs(save_dir, exist_ok=True)
#         # 使用固定的文件名来确保覆盖
#         plot_filename = f'incremental_learning_fold_{fold_num + 1}.png'
#         save_path = os.path.join(save_dir, plot_filename)
#         plt.savefig(save_path, dpi=300)
#         plt.close()
#         print(f"成功: 增量学习曲线图已保存 -> {save_path}")
#     # ========================================================================
#
#     if mdad_plot_epochs:
#         plt.figure(figsize=(12, 8))
#         plt.plot(mdad_plot_epochs, mdad_plot_aucs, marker='o', label='MDAD AUC')
#         plt.plot(mdad_plot_epochs, mdad_plot_auprs, marker='s', label='MDAD AUPR')
#         plt.title(f'MDAD Performance vs Epoch (Fold {fold_num + 1})', fontsize=16)
#         plt.xlabel('Epoch')
#         plt.ylabel('Score')
#         plt.legend()
#         plt.grid(True)
#         os.makedirs(save_dir, exist_ok=True)
#         mdad_path = os.path.join(save_dir, f'mdad_epoch_curve_fold_{fold_num + 1}.png')
#         plt.savefig(mdad_path, dpi=300)
#         plt.close()
#         print(f"MDAD旧任务曲线已保存至: {mdad_path}")
#
#
#     return model, decoder



# 替换 train_eval.py 中旧的 train_gcn_ewc_new 函数
def train_gcn_ewc_new(
        # --- DrugVirus (新任务) 数据 ---
        train_data, edge_index, edge_weight, drug_fg, drug_features, drug_bert,
        microbe_features, microbe_bert, microbe_path, microbe_offset,

        # --- MDAD (旧任务) 数据 ---
        mdad_test_data, mdad_train_data, mdad_edge_index, mdad_edge_weight,
        mdad_drug_fg_norm, mdad_drug_features_norm, mdad_drug_bert_norm,
        mdad_microbe_features_norm, mdad_microbe_bert_norm, mdad_microbe_path_norm,
        mdad_microbe_offset,

        # --- 训练超参数 ---
        epochs, lr, hidden, dropout, device,

        # --- EWC 相关参数 ---
        old_params, old_params_decoder, fisher, fisher_decoder, lambda_ewc,

        # --- 其他 ---
        model=None, decoder=None, args=None,

        # --- 【核心修改】: 接收对齐MLP模块 ---
        alignment_mlps=None,

        # ========== 【在这里新增】接收特征对齐开关 ==========
        use_feature_alignment=True,

        # --- 任务权重超参数 ---
        alpha=0.5,
        weight_decay=1e-5,

        # --- 绘图相关参数 ---
        drugvirus_test_data=None,
        fold_num=0,
        save_dir='.'


):

    """
    【修改版】支持EWC、多任务排练，并与特征对齐MLP进行端到端训练的GCN函数。
    """
    model = model.to(device)
    decoder = decoder.to(device)



    # ========================= 【在此处添加】 =========================
    # 将传入的内部对齐MLP附加到主模型对象上，方便后续在外部调用
    if alignment_mlps is not None:
        model.alignment_mlps = alignment_mlps.to(device)
    else:
        model.alignment_mlps = None
    # =================================================================




    # # --- 【核心修改】: 将对齐MLP的参数加入优化器 ---
    # params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters())) + \
    #                      list(filter(lambda p: p.requires_grad, decoder.parameters()))
    # if alignment_mlps is not None:
    #     alignment_mlps.train()  # 确保它在训练模式
    #     params_to_optimize += list(alignment_mlps.parameters())
    #     print("信息: 对齐MLP的参数已加入优化器，将进行端到端训练。")
    # else:
    #     print("警告: 未提供对齐MLP (alignment_mlps=None)，将直接使用原始DrugVirus特征。")
    # train_eval.py, ~1000行 (替换后的代码)

    # train_eval.py, ~1000行 (替换成这个修正版)

    # --- 【核心修改】: 为不同模块设置独立的学习率 (已修复参数重复问题) ---
    # 1. 定义一个新的学习率参数，专门给对齐MLP用
    lr_align = 0.00001  # <--- 在这里设置对齐MLP的专属学习率.0.00001

    # 2. 准备一个列表来存放所有的参数组
    params_groups = []
    main_model_params = []  # 用来存放不包含对齐MLP的主模型参数

    if alignment_mlps is not None and use_feature_alignment:
        # 如果启用对齐，我们需要将主模型的参数和对齐MLP的参数分开
        alignment_mlps.train()

        # 获取对齐MLP参数的ID，用于后续过滤
        align_param_ids = set(id(p) for p in alignment_mlps.parameters())

        # 过滤出不属于对齐MLP的主模型参数
        main_model_params = [p for p in model.parameters() if id(p) not in align_param_ids and p.requires_grad]

        # 为对齐MLP创建独立的参数组
        params_groups.append({
            'params': alignment_mlps.parameters(),
            'lr': lr_align  # <--- 使用专属学习率
        })
        print(f"信息: 特征对齐已启用，主模型LR={lr}, 对齐MLP LR={lr_align}。")
    else:
        # 如果不启用对齐，所有模型参数都使用主学习率
        main_model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if alignment_mlps is not None and not use_feature_alignment:
            print("信息: 特征对齐已禁用，将不训练对齐MLP并使用原始DrugVirus特征。")
        else:
            print("警告: 未提供对齐MLP (alignment_mlps=None)，将直接使用原始DrugVirus特征。")

    # 3. 将主模型（已过滤）和解码器的参数加入参数组列表
    params_groups.append({
        'params': main_model_params,
        'lr': lr  # 使用主学习率
    })
    params_groups.append({
        'params': filter(lambda p: p.requires_grad, decoder.parameters()),
        'lr': lr  # 使用主学习率
    })

    # 4. 将参数组列表传给优化器
    optimizer = torch.optim.Adam(
        params_groups,  # <--- 传入修复后的参数组列表
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    criterion = nn.BCEWithLogitsLoss()
    model.train()
    decoder.train()

    # --- 一次性转换 MDAD 特征为 Tensor (保持不变) ---
    mdad_drug_fg_norm_t = torch.tensor(mdad_drug_fg_norm, dtype=torch.float32, device=device)
    mdad_drug_features_norm_t = torch.tensor(mdad_drug_features_norm, dtype=torch.float32, device=device)
    mdad_drug_bert_norm_t = torch.tensor(mdad_drug_bert_norm, dtype=torch.float32, device=device)
    mdad_microbe_features_norm_t = torch.tensor(mdad_microbe_features_norm, dtype=torch.float32, device=device)
    mdad_microbe_bert_norm_t = torch.tensor(mdad_microbe_bert_norm, dtype=torch.float32, device=device)
    mdad_microbe_path_norm_t = torch.tensor(mdad_microbe_path_norm, dtype=torch.float32, device=device)

    print_once_set = set()

    # 初始化绘图列表
    plot_epochs, plot_aucs, plot_auprs = [], [], []
    mdad_plot_epochs, mdad_plot_aucs, mdad_plot_auprs = [], [], []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- 1. DrugVirus (新任务) 前向传播和损失计算 ---
        drug_idx, microbe_idx, labels = train_data

        # --- 【核心修改】: 在循环内进行特征对齐 ---
        # 将原始DrugVirus特征转为Tensor
        drug_fg_raw_t = torch.tensor(drug_fg, dtype=torch.float32, device=device)
        drug_features_raw_t = torch.tensor(drug_features, dtype=torch.float32, device=device)
        drug_bert_raw_t = torch.tensor(drug_bert, dtype=torch.float32, device=device)
        microbe_features_raw_t = torch.tensor(microbe_features, dtype=torch.float32, device=device)
        microbe_bert_raw_t = torch.tensor(microbe_bert, dtype=torch.float32, device=device)
        microbe_path_raw_t = torch.tensor(microbe_path, dtype=torch.float32, device=device)

        drugvirus_raw_feats = [
            drug_fg_raw_t, drug_features_raw_t, drug_bert_raw_t,
            microbe_features_raw_t, microbe_bert_raw_t, microbe_path_raw_t
        ]

        # 如果有对齐MLP，则应用变换；否则直接使用原始特征
        # if alignment_mlps is not None:
        #     aligned_feats = [
        #         alignment_mlps[i](feat) for i, feat in enumerate(drugvirus_raw_feats)
        #     ]
        # else:
        #     aligned_feats = drugvirus_raw_feats
        # 如果有对齐MLP且开关为True，则应用变换；否则直接使用原始特征
        if alignment_mlps is not None and use_feature_alignment:
            aligned_feats = [
                alignment_mlps[i](feat) for i, feat in enumerate(drugvirus_raw_feats)
            ]
        else:
            aligned_feats = drugvirus_raw_feats

        # 将对齐后的特征传递给模型
        embeddings_dv, _ = model(
            tuple(aligned_feats),  # 传入对齐后的特征元组
            edge_index, edge_weight
        )
        drug_emb_dv = embeddings_dv[drug_idx]
        microbe_emb_dv = embeddings_dv[microbe_offset + microbe_idx]
        logits_dv = decoder(drug_emb_dv, microbe_emb_dv)
        loss_drugvirus = criterion(logits_dv, torch.tensor(labels, dtype=torch.float32, device=device))

        # --- 2. MDAD (旧任务) 前向传播和损失计算 (保持不变) ---
        mdad_drug_idx, mdad_microbe_idx, mdad_labels = mdad_train_data
        embeddings_mdad, _ = model(
            (mdad_drug_fg_norm_t, mdad_drug_features_norm_t, mdad_drug_bert_norm_t,
             mdad_microbe_features_norm_t, mdad_microbe_bert_norm_t, mdad_microbe_path_norm_t),
            mdad_edge_index, mdad_edge_weight
        )
        drug_emb_mdad = embeddings_mdad[mdad_drug_idx]
        microbe_emb_mdad = embeddings_mdad[mdad_microbe_offset + mdad_microbe_idx]
        logits_mdad = decoder(drug_emb_mdad, microbe_emb_mdad)
        loss_mdad = criterion(logits_mdad, torch.tensor(mdad_labels, dtype=torch.float32, device=device))

        # --- 3. EWC 损失计算 (保持不变) ---
        ewc_loss = ewc_loss_fn(model, decoder, old_params, old_params_decoder,
                               fisher, fisher_decoder, lambda_ewc, print_once_set)

        # --- 4. 构建最终的总损失 (保持不变) ---
        total_loss = (1 - alpha) * loss_drugvirus + alpha * loss_mdad + ewc_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # --- 定期评估并记录数据 ---
        if (epoch + 1) % 6 == 0 and drugvirus_test_data is not None:
            model.eval()
            decoder.eval()
            if alignment_mlps: alignment_mlps.eval()

            with torch.no_grad():
                # --- 【核心修改】: 评估前也需要对齐特征 ---
                if alignment_mlps is not None:
                    aligned_feats_eval = [mlp(feat).cpu().numpy() for mlp, feat in zip(alignment_mlps, drugvirus_raw_feats)]
                    drug_fg_eval, drug_features_eval, drug_bert_eval, \
                    microbe_features_eval, microbe_bert_eval, microbe_path_eval = aligned_feats_eval
                else: # 如果没有MLP，直接用原始numpy数组
                    drug_fg_eval, drug_features_eval, drug_bert_eval, \
                    microbe_features_eval, microbe_bert_eval, microbe_path_eval = \
                    drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path

                # 评估 DrugVirus 性能
                # test_auc, test_aupr = evaluate_gcn(
                #     model, decoder, drugvirus_test_data, edge_index, edge_weight,
                #     drug_fg_eval, drug_features_eval, drug_bert_eval,
                #     microbe_features_eval, microbe_bert_eval, microbe_path_eval,
                #     microbe_offset, device
                # )
                # 评估 DrugVirus 性能
                test_auc, test_aupr, test_acc = evaluate_gcn(
                    model, decoder, drugvirus_test_data, edge_index, edge_weight,
                    drug_fg_eval, drug_features_eval, drug_bert_eval,
                    microbe_features_eval, microbe_bert_eval, microbe_path_eval,
                    microbe_offset, device
                )

                plot_epochs.append(epoch + 1)
                plot_aucs.append(test_auc)
                plot_auprs.append(test_aupr)

                # 评估 MDAD 性能
                # mdad_auc, mdad_aupr = evaluate_gcn(
                #     model, decoder, mdad_test_data, mdad_edge_index, mdad_edge_weight,
                #     mdad_drug_fg_norm, mdad_drug_features_norm, mdad_drug_bert_norm,
                #     mdad_microbe_features_norm, mdad_microbe_bert_norm, mdad_microbe_path_norm,
                #     mdad_microbe_offset, device
                # )
                # 评估 MDAD 性能
                # 评估 MDAD 性能
                mdad_auc, mdad_aupr, mdad_acc = evaluate_gcn(
                    model, decoder, mdad_test_data, mdad_edge_index, mdad_edge_weight,
                    mdad_drug_fg_norm, mdad_drug_features_norm, mdad_drug_bert_norm,
                    mdad_microbe_features_norm, mdad_microbe_bert_norm, mdad_microbe_path_norm,
                    mdad_microbe_offset, device
                )

                mdad_plot_epochs.append(epoch + 1)
                mdad_plot_aucs.append(mdad_auc)
                mdad_plot_auprs.append(mdad_aupr)

            model.train()
            decoder.train()
            if alignment_mlps: alignment_mlps.train()

        if (epoch + 1) % 40 == 0:
            print(
                f"[EWC-MTL Fold {fold_num + 1}] Epoch {epoch + 1}/{epochs}, "
                f"Loss_DV: {loss_drugvirus.item():.4f}, "
                f"Loss_MDAD: {loss_mdad.item():.4f}, "
                f"Loss_EWC: {ewc_loss.item():.4f}, "
                f"Total: {total_loss.item():.4f}"
            )

    # --- 训练结束后绘图并保存 (逻辑不变) ---
    if plot_epochs:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 8))
        plt.plot(plot_epochs, plot_aucs, marker='o', linestyle='-', label='DrugVirus Test AUC')
        plt.plot(plot_epochs, plot_auprs, marker='s', linestyle='--', label='DrugVirus Test AUPR')
        plt.title(f'Incremental Learning Curve (Fold {fold_num + 1})', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = f'incremental_learning_fold_{fold_num + 1}.png'
        save_path = os.path.join(save_dir, plot_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"成功: 增量学习曲线图已保存 -> {save_path}")

    if mdad_plot_epochs:
        plt.figure(figsize=(12, 8))
        plt.plot(mdad_plot_epochs, mdad_plot_aucs, marker='o', label=f'{args.dataset} AUC')
        plt.plot(mdad_plot_epochs, mdad_plot_auprs, marker='s', label=f'{args.dataset} AUPR')
        plt.title(f'{args.dataset} Performance vs Epoch (Fold {fold_num + 1})', fontsize=16)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        mdad_path = os.path.join(save_dir, f'{args.dataset.lower()}_epoch_curve_fold_{fold_num + 1}.png')
        plt.savefig(mdad_path, dpi=300)
        plt.close()
        print(f"{args.dataset}旧任务曲线已保存至: {mdad_path}")

    return model, decoder


# =================================================================================
#  请将此新函数添加到 train_eval.py 文件中（可以放在文件靠前的位置）
# =================================================================================
def pretrain_alignment_mlp_by_stats(
        source_feats,
        target_feats,
        device,
        epochs=150,
        lr=0.005
):
    """
    【新增函数】
    为增量学习预训练特征对齐MLP。
    目标：将 source_feats (DrugVirus) 通过MLP变换后，使其输出的全局均值和标准差
          与 target_feats (MDAD) 的全局均值和标准差相匹配。
    这个函数是独立的，只在增量学习前调用一次。
    """
    import torch.nn as nn
    import torch.optim as optim

    # 1. 创建MLP列表，每个特征一个
    mlp_list = []
    for src_feat, tgt_feat in zip(source_feats, target_feats):
        in_dim = src_feat.shape[1]
        out_dim = tgt_feat.shape[1]
        # 定义一个简单的MLP来进行维度变换和分布对齐
        mlp = nn.Sequential(
            nn.Linear(in_dim, (in_dim + out_dim) // 2),
            nn.ReLU(),
            nn.Linear((in_dim + out_dim) // 2, out_dim)
        ).to(device)
        mlp_list.append(mlp)

    # 将MLP列表封装成nn.ModuleList，以便优化器能识别所有参数
    alignment_mlps = nn.ModuleList(mlp_list)
    optimizer = optim.Adam(alignment_mlps.parameters(), lr=lr)

    # 2. 计算目标特征（MDAD）的均值和标准差（只需计算一次）
    target_means = [torch.tensor(t.mean(axis=0), dtype=torch.float32, device=device) for t in target_feats]
    target_stds = [torch.tensor(t.std(axis=0), dtype=torch.float32, device=device) for t in target_feats]

    # 将源特征（DrugVirus）转为Tensor
    source_tensors = [torch.tensor(s, dtype=torch.float32, device=device) for s in source_feats]

    print("===== 开始为增量学习预训练特征对齐MLP (基于全局统计量) =====")
    alignment_mlps.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0

        # 3. 对每一种特征，计算其变换后的分布与目标分布的差距
        for i, mlp in enumerate(alignment_mlps):
            # 将源特征通过MLP进行变换
            predicted_feat = mlp(source_tensors[i])

            # 计算变换后特征的均值和标准差
            pred_mean = predicted_feat.mean(dim=0)
            pred_std = predicted_feat.std(dim=0)

            # 损失函数 = 均值MSE + 标准差MSE
            loss = nn.functional.mse_loss(pred_mean, target_means[i]) + \
                   nn.functional.mse_loss(pred_std, target_stds[i])
            total_loss += loss

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"[对齐预训练] Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.6f}")

    print("===== 特征对齐MLP预训练完成 =====")
    alignment_mlps.eval()  # 切换到评估模式
    return alignment_mlps


# train_eval.py 文件中

import torch.nn as nn
import torch.optim as optim


def pretrain_alignment_mlp_by_stats_v2(
        source_feats,
        target_feats,
        device,
        epochs=150,
        lr=0.005
):
    """
    【改进版】
    为增量学习预训练特征对齐MLP。
    结合了【正交初始化】和【BatchNorm1d】，使训练更稳定高效。
    """
    mlp_list = []
    for src_feat, tgt_feat in zip(source_feats, target_feats):
        in_dim = src_feat.shape[1]
        out_dim = tgt_feat.shape[1]

        # --- 核心修改：改进MLP结构并应用更好的初始化 ---
        mlp = nn.Sequential(
            nn.Linear(in_dim, (in_dim + out_dim) // 2),
            nn.BatchNorm1d((in_dim + out_dim) // 2),  # 添加BatchNorm层
            nn.ReLU(),
            nn.Linear((in_dim + out_dim) // 2, out_dim)
        ).to(device)

        # --- 对线性层应用正交初始化 ---
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)  # 应用正交初始化
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # 偏置初始化为0

        mlp_list.append(mlp)

    alignment_mlps = nn.ModuleList(mlp_list)
    optimizer = optim.Adam(alignment_mlps.parameters(), lr=lr)

    target_means = [torch.tensor(t.mean(axis=0), dtype=torch.float32, device=device) for t in target_feats]
    target_stds = [torch.tensor(t.std(axis=0), dtype=torch.float32, device=device) for t in target_feats]
    source_tensors = [torch.tensor(s, dtype=torch.float32, device=device) for s in source_feats]

    print("===== 开始为增量学习预训练特征对齐MLP (V2: 正交初始化 + BatchNorm) =====")
    alignment_mlps.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0

        for i, mlp in enumerate(alignment_mlps):
            predicted_feat = mlp(source_tensors[i])
            pred_mean = predicted_feat.mean(dim=0)
            pred_std = predicted_feat.std(dim=0)

            loss = nn.functional.mse_loss(pred_mean, target_means[i]) + \
                   nn.functional.mse_loss(pred_std, target_stds[i])
            total_loss += loss

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"[对齐预训练 V2] Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.6f}")

    print("===== 特征对齐MLP预训练完成 (V2) =====")
    alignment_mlps.eval()
    return alignment_mlps


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


def _ensure_tensor_concat(feat_block, device):
    """
    Accept numpy / tensor / list(tuple) of them, flatten to single tensor on device.
    """
    if isinstance(feat_block, (np.ndarray, torch.Tensor)):
        tensors = [feat_block]
    elif isinstance(feat_block, (list, tuple)):
        tensors = feat_block
    else:
        raise TypeError(f"Unsupported feature container type: {type(feat_block)}")

    torch_list = []
    for item in tensors:
        if isinstance(item, np.ndarray):
            torch_list.append(torch.from_numpy(item))
        elif torch.is_tensor(item):
            torch_list.append(item.detach().cpu())
        else:
            raise TypeError(f"Unsupported feature element type: {type(item)}")
    return torch.cat(torch_list, dim=0).to(device=device, dtype=torch.float32)


def _torch_cov(x):
    x = x - x.mean(dim=0, keepdim=True)
    return x.t() @ x / (x.size(0) - 1)


def _rbf_kernel(x, y, gamma):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2 * x @ y.t()
    return torch.exp(-gamma * dist.clamp_min_(0.0))


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


def _ensure_tensor_concat(block, device):
    if isinstance(block, (np.ndarray, torch.Tensor)):
        items = [block]
    elif isinstance(block, (list, tuple)):
        items = block
    else:
        raise TypeError(f"Unsupported type: {type(block)}")

    tensors = []
    for item in items:
        if isinstance(item, np.ndarray):
            tensors.append(torch.from_numpy(item))
        elif torch.is_tensor(item):
            tensors.append(item.detach().cpu())
        else:
            raise TypeError(f"Unsupported element type: {type(item)}")
    return torch.cat(tensors, dim=0).to(device=device, dtype=torch.float32)


def _torch_cov(x):
    x = x - x.mean(dim=0, keepdim=True)
    return x.t() @ x / (x.size(0) - 1)


def _rbf_kernel(x, y, gamma):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2 * x @ y.t()
    return torch.exp(-gamma * dist.clamp_min_(0.0))


# def pretrain_alignment_mlp_by_stats_v3(
#         source_feats,
#         target_feats,
#         device,
#         epochs: int = 400,
#         lr: float = 5e-4,
#         hidden_ratio: float = 0.5,
#         proj_hidden_ratio: float = 0.75,
#         mmd_gamma: float = 0.01,
#         rand_pair_weight: float = 0.1,
#         mmd_weight: float = 0.3,
#         cov_weight: float = 0.5,
#         mean_weight: float = 0.5,
#         std_weight: float = 0.5,
#         verbose: bool = True,
# ):
#     """
#     维度可不同；平均值/方差/协方差/MMD/随机配对多重约束。
#     输入可为 numpy、tensor 或 list/tuple。
#     返回 nn.ModuleList。
#     """
#     if len(source_feats) != len(target_feats):
#         raise ValueError("source_feats 与 target_feats 的长度必须一致。")
#
#     mlp_list = nn.ModuleList().to(device)
#     optimizer_params = []
#
#     for idx, (src_block, tgt_block) in enumerate(zip(source_feats, target_feats)):
#         src_all = _ensure_tensor_concat(src_block, device)
#         tgt_all = _ensure_tensor_concat(tgt_block, device)
#
#         in_dim = src_all.size(1)
#         out_dim = tgt_all.size(1)
#         hidden_dim = max(8, int(in_dim * hidden_ratio))
#         proj_hidden = max(8, int((in_dim + out_dim) * proj_hidden_ratio * 0.5))
#
#         mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, proj_hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(proj_hidden, out_dim)
#         ).to(device)
#
#         nn.init.orthogonal_(mlp[0].weight); nn.init.zeros_(mlp[0].bias)
#         nn.init.orthogonal_(mlp[2].weight); nn.init.zeros_(mlp[2].bias)
#         nn.init.orthogonal_(mlp[4].weight); nn.init.zeros_(mlp[4].bias)
#
#         mlp_list.append(mlp)
#         optimizer_params += list(mlp.parameters())
#
#     optimizer = Adam(optimizer_params, lr=lr, weight_decay=1e-5)
#     loss_log = []
#
#     iterator = tqdm(range(epochs), desc="Align v4") if verbose else range(epochs)
#
#     with torch.no_grad():
#         target_stats = []
#         for tgt_block in target_feats:
#             tgt_all = _ensure_tensor_concat(tgt_block, device)
#             target_stats.append({
#                 "mean": tgt_all.mean(dim=0),
#                 "std": tgt_all.std(dim=0),
#                 "cov": _torch_cov(tgt_all),
#                 "samples": tgt_all
#             })
#
#     for epoch in iterator:
#         optimizer.zero_grad()
#         total_loss = 0.0
#
#         for feat_idx, (mlp, tgt_stat) in enumerate(zip(mlp_list, target_stats)):
#             src_all = _ensure_tensor_concat(source_feats[feat_idx], device)
#             transformed = mlp(src_all)
#
#             mean_loss = F.mse_loss(transformed.mean(dim=0), tgt_stat["mean"])
#             std_loss = F.mse_loss(transformed.std(dim=0), tgt_stat["std"])
#             cov_loss = F.mse_loss(_torch_cov(transformed), tgt_stat["cov"])
#
#             k_xx = _rbf_kernel(transformed, transformed, mmd_gamma).mean()
#             k_yy = _rbf_kernel(tgt_stat["samples"], tgt_stat["samples"], mmd_gamma).mean()
#             k_xy = _rbf_kernel(transformed, tgt_stat["samples"], mmd_gamma).mean()
#             mmd_loss = k_xx + k_yy - 2 * k_xy
#
#             rand_idx = torch.randint(0, tgt_stat["samples"].size(0), (transformed.size(0),), device=device)
#             pair_loss = F.l1_loss(transformed, tgt_stat["samples"][rand_idx])
#
#             loss = (mean_weight * mean_loss +
#                     std_weight * std_loss +
#                     cov_weight * cov_loss +
#                     mmd_weight * mmd_loss +
#                     rand_pair_weight * pair_loss)
#             total_loss += loss
#
#         total_loss.backward()
#         optimizer.step()
#
#         if verbose:
#             loss_log.append(total_loss.item())
#             iterator.set_postfix(loss=f"{total_loss.item():.4f}")
#
#     if verbose and loss_log:
#         print(f"[Align v4] Final loss: {loss_log[-1]:.6f}")
#
#     return mlp_list


# In your utils.py or equivalent file

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Helper function to compute covariance matrix
def torch_cov(tensor, t=None):
    """
    Computes the covariance matrix of a 2D tensor.
    tensor: (n_samples, n_features)
    """
    if t is None:
        t = tensor
    # Subtract the mean
    tensor_mean = torch.mean(tensor, dim=0, keepdim=True)
    t_mean = torch.mean(t, dim=0, keepdim=True)

    # (n_features, n_samples) x (n_samples, n_features) -> (n_features, n_features)
    cov = (tensor - tensor_mean).t() @ (t - t_mean) / (tensor.size(0) - 1)
    return cov

