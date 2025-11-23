import warnings
warnings.filterwarnings("ignore")
# main.py
# ... (其他 import) ...
from gcn_model import GCNWithDecoderWrapper, explain_with_full_graph, explain_with_khop_subgraph
# ======================== 【在这里新增】 ========================
from aggregate_visualization import run_aggregate_visualization
# ===============================================================
import pandas as pd
# ...

import torch
from negative_sampler import sel_neg_by_bagging
from sklearn.model_selection import KFold # 我们用KFold来切分样本，更标准
from visualization import *
# ========== 每次运行脚本前先清空一次CUDA显存 ==========
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("已清空本地CUDA显存")
# =================================================

#from torch_geometric.nn import PGExplainer
from torch_geometric.explain.algorithm import PGExplainer
from custom_gradcam import CustomGradCAM
import gc
print("PGExplainer类来源:", PGExplainer)
import torch_geometric
print(f'pyg的版本：{torch_geometric.__version__}')
# diagnostic_script.py
#from torch_geometric.explain.algorithm import PGExplainer,GradCAM
try:
    from torch_geometric.explain.config import ModelReturnType

    print("PyG环境中所有可用的 ModelReturnType 成员:")
    print("=" * 40)
    # 使用 __members__ 属性来遍历所有枚举成员
    for member_name in ModelReturnType.__members__:
        print(f"- {member_name}")
    print("=" * 40)

except ImportError:
    print("无法导入 ModelReturnType。请检查 torch_geometric 是否已正确安装。")
except Exception as e:
    print(f"发生错误: {e}")

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cuda is available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else "None")
print("device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
# 1. 导入PGExplainer

from data_utils import *
from train_eval import train_gcn, evaluate_gcn
from train_eval import * # 加载新函数
from sklearn.model_selection import train_test_split # <--- 新增此行
# main.py 头部（import区域）添加
from train_eval import compute_fisher_gcn

from gcn_model import GCNWithDecoderWrapper, explain_with_full_graph, explain_with_khop_subgraph

import pandas as pd
from gcn_model import normalize_adj

import argparse
import numpy as np
from train_eval import pretrain_alignment_mlp_by_stats
import networkx as nx
import matplotlib.pyplot as plt
# fold loop之前
import os
from gcn_model import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
import matplotlib.pyplot as plt
import matplotlib
import matplotlib
#matplotlib.use('TkAgg')
from torch_geometric.nn import GCNConv
from gcn_model import *
# 设置支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
#device = torch.device('cpu')
# 动态拼模型和解码器保存路径，包含数据集和fold编号
def get_model_path(dataset, fold):
    return f'./{dataset}/{dataset}_gcn_model_fold{fold+1}.pth'
def get_decoder_path(dataset, fold):
    return f'./{dataset}/{dataset}_decoder_fold{fold+1}.pth'

# 添加命令行参数解析
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='MDAD', choices=['DrugVirus', 'MDAD','aBiofilm'],
                   help="Dataset to use: MDAD or DrugVirus or aBiofilm")
parser.add_argument('--epochs', type=int, default=400, help="Number of training epochs")#700
parser.add_argument('--learning_rate', type=float, default=0.02, help="Learning rate")#0.02
parser.add_argument('--wd', type=float, default=0.0, help="Weight decay (L2 penalty) for initial training")
#
parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")#0.5
parser.add_argument('--retrain', action='store_true', help='强制重新训练模型')

# --- 【新增】增强邻接矩阵后，重训练的超参数 ---
parser.add_argument('--epochs_retrain', type=int, default=100, help="Number of retraining epochs for the fused graph.")#800
parser.add_argument('--lr_retrain', type=float, default=0.02, help="Learning rate for retraining on the fused graph.")
parser.add_argument('--wd_retrain', type=float, default=1e-4, help="Weight decay for retraining on the fused graph.")#-4
#
parser.add_argument('--dropout_retrain', type=float, default=0.4, help="Dropout rate for retraining on the fused graph.")
parser.add_argument('--hidden_dim_retrain', type=float, default=64, help="Dropout rate for retraining on the fused graph.")#64


# ======================== 【在这里新增】 ========================
parser.add_argument('--lr_step_size', type=int, default=200, help="Step size for learning rate scheduler (how many epochs between decay)")
parser.add_argument('--lr_gamma', type=float, default=0.9, help="Decay factor for learning rate scheduler")
# ==

# --- 【新增】早停参数 ---
parser.add_argument('--early_stopping_patience', type=int, default=50,
                    help="Patience for early stopping. Set to 0 to disable.")
# ===============================================================
# ========== 【在这里新增】增量学习特征对齐开关 ==========
parser.add_argument('--no_feature_alignment', action='store_true', default=False,
                    help='在增量学习阶段，禁用从DrugVirus到MDAD的特征对齐（默认禁用）。')#False是启用mlp
# ==================== 【在这里新增开关】 ====================
parser.add_argument('--use', type=bool, default=False,
                    help='如果设置为True，则直接读取已保存的重要性矩阵，跳过GNNExplainer的耗时计算')#（默认为False,False是# ）
# ======================== 【在这里新增】 ========================
parser.add_argument('--incremental_dataset', type=str, default='MDAD', choices=['DrugVirus', 'MDAD', 'aBiofilm'],
                   help="用于增量学习的数据集。如果不指定，则跳过此步骤。")
#

args = parser.parse_args()

print(f"Selected dataset: {args.dataset}")


if args.dataset == 'MDAD':
    base_path = './MDAD/'

    DRUG_FEATURE_PATH = base_path + 'drugfeatures.txt'
    DRUG_BERT_PATH=base_path + 'drug_bert.xlsx'
    DRUG_FG_PATH=base_path + 'fingerprint.xlsx'

    #MICROBE_FEATURE_PATH = base_path + 'microbe(name+fm特征.txt'
    MICROBE_FEATURE_PATH=base_path + 'microbefeatures.txt'
    MICROBE_BERT_PATH = base_path + 'microbe_bert.xlsx'
    MICROBE_PATH_PATH=base_path + 'microbe_path.xlsx'

    DRUG_SIM_PATH = base_path + 'drugsimilarity.txt'
    MICROBE_SIM_PATH = base_path + 'microbesimilarity.txt'
    args.lr_retrain = 0.001  # 提高学习率
    ADJ_PATH = base_path + 'adj_out.txt'
    microbe_offset = 1373  # MDAD数据集的微生物偏移量
elif args.dataset == 'DrugVirus':
    base_path = './DrugVirus/'

    DRUG_FEATURE_PATH = base_path + 'drugfeatures.txt'
    DRUG_BERT_PATH = base_path + 'drug_bert.xlsx'
    DRUG_FG_PATH = base_path + 'fingerprint.xlsx'

    #MICROBE_FEATURE_PATH = base_path + 'microbe(name+fm特征.txt'
    MICROBE_FEATURE_PATH = base_path + 'microbefeatures.txt'
    MICROBE_BERT_PATH = base_path + 'microbe_bert.xlsx'
    MICROBE_PATH_PATH = base_path + 'microbe_path.xlsx'

    DRUG_SIM_PATH = base_path + 'drugsimilarity.txt'
    MICROBE_SIM_PATH = base_path + 'microbesimilarity.txt'
    ADJ_PATH = base_path + 'adj_out.txt'
    microbe_offset = 175  # DrugVirus数据集的微生物偏移量


    # #专门为DrugVirus设置超参（在420行附近）：
    # if args.dataset == 'DrugVirus':
    args.lr_retrain = 0.001 # 提高学习率/0.001
    args.learning_rate=0.002
    args.epochs_retrain = 300  # 增加训练轮数
    #     args.epochs_retrain=500
    #     args.hidden_dim = 256  # 增加隐藏层维度
    args.hidden_dim_retrain = 64  # 增加隐藏层维度
    args.dropout=0.4
    args.dropout_retrain = 0.4  # 降低dropout率
elif args.dataset == 'aBiofilm':
    base_path = './aBiofilm/'

    DRUG_FEATURE_PATH = base_path + 'drugfeatures.txt'
    DRUG_BERT_PATH = base_path + 'drug_bert.xlsx'
    DRUG_FG_PATH = base_path + 'fingerprint.xlsx'

    #MICROBE_FEATURE_PATH = base_path + 'microbe(name+fm特征.txt'
    MICROBE_FEATURE_PATH = base_path + 'microbefeatures.txt'
    MICROBE_BERT_PATH = base_path + 'microbe_bert.xlsx'
    MICROBE_PATH_PATH = base_path + 'microbe_path.xlsx'

    DRUG_SIM_PATH = base_path + 'drugsimilarity.txt'
    MICROBE_SIM_PATH = base_path + 'microbesimilarity.txt'
    ADJ_PATH = base_path + 'adj_out.txt'
    microbe_offset = 1720  # aBiofilm数据集的微生物偏移量

    args.lr_retrain = 0.01  # 提高学习率/0.001
# 读取特征和邻接矩阵
print('加载特征和邻接矩阵...')
#drug_features, microbe_features = load_features(DRUG_FEATURE_PATH, MICROBE_FEATURE_PATH)
drug_features,drug_bert,drug_fg,microbe_features,microbe_bert,microbe_path=load_features(DRUG_FEATURE_PATH,DRUG_BERT_PATH,DRUG_FG_PATH,MICROBE_FEATURE_PATH,MICROBE_BERT_PATH,MICROBE_PATH_PATH)
#microbe_features = np.loadtxt(MICROBE_FEATURE_PATH)
#df = pd.read_csv('D:/a论文sci专属文件夹\数据集处理\处理MDAD\GCN/adj_out.txt', sep='\s+', header=None, names=['drug', 'microbe', 'value'])
df = pd.read_csv(ADJ_PATH, sep='\s+', header=None, names=['drug', 'microbe', 'value'])

num_drug = df['drug'].max()
num_microbe = df['microbe'].max()
adj = np.zeros((num_drug+1, num_microbe+1), dtype=int)
for _, row in df.iterrows():
    adj[row['drug'], row['microbe']] = row['value']

# 提取药物-微生物子矩阵I

I = adj

Sd = np.loadtxt(DRUG_SIM_PATH)
Sm = np.loadtxt(MICROBE_SIM_PATH)
# Sd = np.zeros_like(Sd)
# Sm = np.zeros_like(Sm)
# 构建GCN输入的块结构邻接矩阵和特征矩阵
# A = build_gcn_adj(Sd, Sm, I)
# A = normalize_adj(A)
# #X = build_gcn_features(drug_features, microbe_features)

# 获取正负样本
print('构建正负样本...')
pos, neg = get_positive_negative_samples(adj)

# 5折交叉验证
n_splits = 5
results = []
random_state = 42#42
results_fused = []
start_fold=0




for fold, (train_pos, test_pos, train_neg, test_neg) in enumerate(get_kfold_indices(pos, neg, n_splits=n_splits, random_state=random_state)):
    if fold < start_fold:  # 跳过前三折
        continue
    print(f'-----------------------------------------------------Fold {fold+1}/{n_splits}---------------------------------------------------------------------')
    # 构建训练集和测试集
    # train_pos_labels = np.ones(len(train_pos))
    # train_neg_sample = sample_negatives(train_neg, len(train_pos), random_state=fold)
    # train_neg_labels = np.zeros(len(train_neg_sample))
    # train_samples = np.vstack([train_pos, train_neg_sample])
    # train_labels = np.concatenate([train_pos_labels, train_neg_labels])
    # train_drug_idx = train_samples[:,0].astype(int)
    # train_microbe_idx = train_samples[:,1].astype
    # --- 步骤 1: 【提前】进行特征归一化 (解决NameError) ---
    from sklearn.preprocessing import StandardScaler

    # 临时用 train_pos 获取训练索引，因为 train_drug_idx 还没生成
    temp_train_drug_idx_for_scaler = train_pos[:, 0].astype(int)
    temp_train_microbe_idx_for_scaler = train_pos[:, 1].astype(int)

    scaler_fg = StandardScaler().fit(drug_fg[temp_train_drug_idx_for_scaler])
    scaler_feat = StandardScaler().fit(drug_features[temp_train_drug_idx_for_scaler])
    scaler_bert = StandardScaler().fit(drug_bert[temp_train_drug_idx_for_scaler])
    drug_fg_norm = scaler_fg.transform(drug_fg)
    drug_features_norm = scaler_feat.transform(drug_features)
    drug_bert_norm = scaler_bert.transform(drug_bert)

    scaler_microbe_feat = StandardScaler().fit(microbe_features[temp_train_microbe_idx_for_scaler])
    scaler_microbe_bert = StandardScaler().fit(microbe_bert[temp_train_microbe_idx_for_scaler])
    scaler_microbe_path = StandardScaler().fit(microbe_path[temp_train_microbe_idx_for_scaler])
    microbe_features_norm = scaler_microbe_feat.transform(microbe_features)
    microbe_bert_norm = scaler_microbe_bert.transform(microbe_bert)
    microbe_path_norm = scaler_microbe_path.transform(microbe_path)

    # --- 步骤 2: 条件化负样本采样策略 ---
    train_pos_labels = np.ones(len(train_pos))

    if args.dataset == 'DugVirus':
        print("★★★ 检测到 DrugVirus 数据集，启用 Bagging 负采样策略 ★★★")

        # 2.1 为了获取初始嵌入，先用随机负样本进行短暂预训练
        print("    -> 步骤 1/4: 短暂预训练以获取初始节点嵌入...")
        # 临时图结构 (移除测试集边)
        I_train_temp = I.copy()
        for d, m in test_pos: I_train_temp[d, m] = 0
        A_fold_temp = build_gcn_adj(Sd, Sm, I_train_temp)
        row_t, col_t = np.where(A_fold_temp != 0)
        edge_index_temp = torch.tensor(np.stack([row_t, col_t]), dtype=torch.long, device=device)
        edge_weight_temp = torch.tensor(A_fold_temp[row_t, col_t], dtype=torch.float32, device=device)

        # 临时训练数据
        temp_train_neg_sample = sample_negatives(train_neg, len(train_pos), random_state=fold)
        temp_train_samples = np.vstack([train_pos, temp_train_neg_sample])
        temp_train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(temp_train_neg_sample))])
        temp_train_data = (
        temp_train_samples[:, 0].astype(int), temp_train_samples[:, 1].astype(int), temp_train_labels)

        # 临时模型和训练
        temp_model = GCNWithMLP(
            drug_in_dim=drug_fg.shape[1], drug_out_dim=175, microbe_dim=microbe_features.shape[1],
            microbe_out_dim=95, gcn_hidden=args.hidden_dim, dropout=args.dropout,
            use_microbe_mlp=False, dataset_name=args.dataset
        ).to(device)
        temp_decoder = MLPDecoder(args.hidden_dim).to(device)
        temp_optimizer = torch.optim.Adam(list(temp_model.parameters()) + list(temp_decoder.parameters()), lr=0.001)#0.01
        temp_criterion = torch.nn.BCEWithLogitsLoss()

        features_for_pretrain = (
            torch.tensor(drug_fg_norm, dtype=torch.float32, device=device),
            torch.tensor(drug_features_norm, dtype=torch.float32, device=device),
            torch.tensor(drug_bert_norm, dtype=torch.float32, device=device),
            torch.tensor(microbe_features_norm, dtype=torch.float32, device=device),
            torch.tensor(microbe_bert_norm, dtype=torch.float32, device=device),
            torch.tensor(microbe_path_norm, dtype=torch.float32, device=device),
        )

        temp_model.train()
        pretrain_epochs = 60  # 这里当前是60
        for pre_epoch in range(pretrain_epochs):  # 短暂预训练
            temp_optimizer.zero_grad()
            embeddings, _ = temp_model(features_for_pretrain, edge_index_temp, edge_weight_temp)
            drug_idx, microbe_idx, labels = temp_train_data
            drug_emb = embeddings[drug_idx]
            microbe_emb = embeddings[microbe_offset + microbe_idx]
            logits = temp_decoder(drug_emb, microbe_emb)
            loss = temp_criterion(logits, torch.tensor(labels, dtype=torch.float32).to(device))
            loss.backward()
            temp_optimizer.step()
            if (pre_epoch + 1) % 20 == 0:
                print(f"       预训练 Epoch {pre_epoch + 1}/{pretrain_epochs}, Loss: {loss.item():.4f}")

        # 2.2 获取预训练后的节点嵌入
        print("    -> 步骤 2/4: 成功生成了初始嵌入。")
        temp_model.eval()
        with torch.no_grad():
            initial_embeddings, _ = temp_model(features_for_pretrain, edge_index_temp, edge_weight_temp)
            initial_embeddings_np = initial_embeddings.cpu().numpy()


        # 2.3 使用Bagging策略筛选高质量负样本
        print("    -> 步骤 3/4: 调用 sel_neg_by_bagging 筛选高质量负样本...")
        train_neg_sample = sel_neg_by_bagging(
            pos_ij=train_pos, unlabelled_ij=train_neg, features=initial_embeddings_np,  # <--- 修改此处：feature -> H
            adj_matrix=I, iterate_time=10
        )

        print(f"    -> 步骤 4/4: Bagging 采样完成，得到 {len(train_neg_sample)} 个高质量负样本。")

        # 2.4 清理临时资源
        del temp_model, temp_decoder, temp_optimizer, initial_embeddings, initial_embeddings_np, edge_index_temp, edge_weight_temp
        torch.cuda.empty_cache()

    else:
        # 对于 MDAD 和 aBiofilm 数据集，使用原来的随机采样方法
        print("◇◇◇ 使用标准随机负采样策略 ◇◇◇")
        train_neg_sample = sample_negatives(train_neg, len(train_pos), random_state=fold)

    # --- 步骤 3: 组装最终的训练集和测试集 (这部分代码对两种策略是通用的) ---
    train_neg_labels = np.zeros(len(train_neg_sample))
    train_samples = np.vstack([train_pos, train_neg_sample])
    train_labels = np.concatenate([train_pos_labels, train_neg_labels])
    train_drug_idx = train_samples[:, 0].astype(int)
    train_microbe_idx = train_samples[:, 1].astype(int)















    test_pos_labels = np.ones(len(test_pos))
    test_neg_sample = sample_negatives(test_neg, len(test_pos), random_state=fold+50)
    test_neg_labels = np.zeros(len(test_neg_sample))
    test_samples = np.vstack([test_pos, test_neg_sample])
    test_labels = np.concatenate([test_pos_labels, test_neg_labels])
    test_drug_idx = test_samples[:,0].astype(int)
    test_microbe_idx = test_samples[:,1].astype(int)



    # ====== 新增：a药物特征归一化流程 ======(先不做归一化)
    # from sklearn.preprocessing import StandardScaler
    # scaler_fg = StandardScaler().fit(drug_fg[train_drug_idx])
    # scaler_feat = StandardScaler().fit(drug_features[train_drug_idx])
    # scaler_bert = StandardScaler().fit(drug_bert[train_drug_idx])
    # # transform全体drug特征
    # drug_fg_norm = scaler_fg.transform(drug_fg)
    # drug_features_norm = scaler_feat.transform(drug_features)
    # drug_bert_norm = scaler_bert.transform(drug_bert)
    # # ============================
    # # 微生物特征归一化（新增）
    # scaler_microbe_feat = StandardScaler().fit(microbe_features[train_microbe_idx])
    # scaler_microbe_bert = StandardScaler().fit(microbe_bert[train_microbe_idx])
    # scaler_microbe_path = StandardScaler().fit(microbe_path[train_microbe_idx])
    # microbe_features_norm = scaler_microbe_feat.transform(microbe_features)
    # microbe_bert_norm = scaler_microbe_bert.transform(microbe_bert)
    # microbe_path_norm = scaler_microbe_path.transform(microbe_path)

    #====== 新增：a药物特征归一化流程 ======(先不做归一化)
    from sklearn.preprocessing import StandardScaler
    scaler_fg = StandardScaler().fit(drug_fg[train_drug_idx])
    scaler_feat = StandardScaler().fit(drug_features[train_drug_idx])
    scaler_bert = StandardScaler().fit(drug_bert[train_drug_idx])
    # transform全体drug特征
    drug_fg_norm = scaler_fg.transform(drug_fg)
    drug_features_norm = scaler_feat.transform(drug_features)
    drug_bert_norm = scaler_bert.transform(drug_bert)
    # ============================
    # 微生物特征归一化（新增）
    scaler_microbe_feat = StandardScaler().fit(microbe_features[train_microbe_idx])
    scaler_microbe_bert = StandardScaler().fit(microbe_bert[train_microbe_idx])
    scaler_microbe_path = StandardScaler().fit(microbe_path[train_microbe_idx])
    microbe_features_norm = scaler_microbe_feat.transform(microbe_features)
    microbe_bert_norm = scaler_microbe_bert.transform(microbe_bert)
    microbe_path_norm = scaler_microbe_path.transform(microbe_path)

    # if args.dataset == 'DrugVirus':
    #     # 对DrugVirus进行额外的数据增强
    #     from sklearn.decomposition import PCA
    #
    #     # 对药物特征进行PCA降维，保留更多信息
    #     pca_drug = PCA(n_components=175)
    #     drug_fg_norm = pca_drug.fit_transform(drug_fg_norm)
    #     drug_features_norm = pca_drug.fit_transform(drug_features_norm)
    #     drug_bert_norm = pca_drug.fit_transform(drug_bert_norm)

    #
    # drug_fg_norm = drug_fg
    # drug_features_norm = drug_features
    # drug_bert_norm = drug_bert
    # microbe_features_norm = microbe_features
    # microbe_bert_norm = microbe_bert
    # microbe_path_norm = microbe_path
    #



    # train_data = (train_drug_idx, train_microbe_idx, train_labels)
    # test_data = (test_drug_idx, test_microbe_idx, test_labels)
    # ======================== 【核心修改】从训练集中划分出验证集 ========================
    # 1. 确定验证集大小，使其与测试集大小相同
    val_size = len(test_samples)

    # 2. 从原始训练数据中划分出新的训练集和验证集
    #    使用 stratify=train_labels 确保划分后训练集和验证集的正负样本比例与原始训练集一致
    train_samples_new, val_samples, train_labels_new, val_labels = train_test_split(
        train_samples, train_labels,
        test_size=val_size,
        random_state=random_state, # 使用与KFold相同的随机种子保证可复现
        stratify=train_labels      # 保持正负样本比例
    )

    # 3. 重新打包成 train_data, val_data, test_data 元组
    # 新的、更小的训练集
    train_drug_idx_new = train_samples_new[:, 0].astype(int)
    train_microbe_idx_new = train_samples_new[:, 1].astype(int)
    train_data = (train_drug_idx_new, train_microbe_idx_new, train_labels_new)

    # 新的验证集
    val_drug_idx = val_samples[:, 0].astype(int)
    val_microbe_idx = val_samples[:, 1].astype(int)
    val_data = (val_drug_idx, val_microbe_idx, val_labels)

    # 原有的 test_data 保持不变，作为最终的、独立的测试集
    test_data = (test_drug_idx, test_microbe_idx, test_labels)

    print(f"数据划分完毕: 新训练集大小 {len(train_samples_new)}, 验证集大小 {len(val_samples)}, 测试集大小 {len(test_samples)}")
    # ==============================================================================

#    # 从新的、更小的训练集中，提取出正样本，用于后续的 GNNExplainer
    # train_labels_new 是与 train_samples_new 对应的标签
    train_pos_mask = (train_labels_new == 1)
    # train_pos_new 只包含那些真正用于训练的正样本边
    train_pos_new = train_samples_new[train_pos_mask]
    print(f"为 GNNExplainer 准备了 {len(train_pos_new)} 条纯训练集正样本边。")
    # =============================================================

    # ======================== 【核心修正】从训练图中移除测试边和验证边 ========================
    # 这一步至关重要，可以防止模型在消息传递阶段“看到”验证集和测试集的边，避免信息泄露。
    print("正在从训练图邻接矩阵中移除测试集和验证集的正样本边...")
    # ... 后续代码不变 ...

















    # I_train = I.copy()
    # for d, m in test_pos:
    #     I_train[d, m] = 0
    # A_fold = build_gcn_adj(Sd, Sm, I_train)  # 这是稠密加权邻接矩阵
    # # 不要normalize_adj(A_fold)！GCNConv会自动做
    # row, col = np.where(A_fold != 0)
    # edge_index = np.stack([row, col], axis=0)  # shape [2, num_edges]
    # edge_weight = A_fold[row, col]  # shape [num_edges]
    # edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    # edge_weight = torch.tensor(edge_weight, dtype=torch.float32, device=device)
    # ======================== 【核心修正】从训练图中移除测试边和验证边 ========================
    # 这一步至关重要，可以防止模型在消息传递阶段“看到”验证集和测试集的边，避免信息泄露。
    print("正在从训练图邻接矩阵中移除测试集和验证集的正样本边...")
    I_train = I.copy()

    # 1. 移除测试集中的正样本边 (这是你已有的正确逻辑)
    for d, m in test_pos:
        I_train[d, m] = 0
    print(f"已从训练图中移除 {len(test_pos)} 条测试边。")

    # 2. 【新增逻辑】找出验证集中的正样本边，然后从训练图中也移除它们
    #    val_samples 和 val_labels 是上一步 train_test_split 的输出
    val_pos_mask = (val_labels == 1)
    val_pos = val_samples[val_pos_mask]
    for d, m in val_pos:
        I_train[d, m] = 0
    print(f"已从训练图中移除 {len(val_pos)} 条验证边。")
    # ======================================================================================


    A_fold = build_gcn_adj(Sd, Sm, I_train)  # 使用处理干净的 I_train 构建图
    # GCNConv会自动对邻接矩阵进行归一化，所以这里不需要手动调用 normalize_adj
    row, col = np.where(A_fold != 0)
    edge_index = np.stack([row, col], axis=0)  # shape [2, num_edges]
    edge_weight = A_fold[row, col]  # shape [num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32, device=device)
















    # ==================== 【第1处修改：保存所有增量学习所需数据】 ====================
    fold_dir_for_save = os.path.join(base_path, f'fold{fold + 1}')
    os.makedirs(fold_dir_for_save, exist_ok=True)
    train_data_save_path = os.path.join(fold_dir_for_save, f'{args.dataset}_fold_data_for_ewc.npz')  # <-- 文件名改得更清晰

    # 保存训练样本、完整的图邻接矩阵、以及所有归一化后的特征
    np.savez(train_data_save_path,
             # 训练样本
             drug_idx=train_data[0],
             microbe_idx=train_data[1],
             labels=train_data[2],
             # 图结构
             adj_matrix=A_fold,  # <-- 【核心新增】保存图的邻接矩阵
             # 特征
             drug_fg_norm=drug_fg_norm,
             drug_features_norm=drug_features_norm,
             drug_bert_norm=drug_bert_norm,
             microbe_features_norm=microbe_features_norm,
             microbe_bert_norm=microbe_bert_norm,
             microbe_path_norm=microbe_path_norm
             )
    print(f"为增量学习做准备: 当前fold的完整数据已保存至 -> {train_data_save_path}")
    # ==============================================================================




    # === 新增：保存这一折的测试集到本地，方便微调后评估用 ===
    np.savez(f'./{args.dataset}/fold{fold + 1}/{args.dataset}_test_set.npz',
             test_drug_idx=test_drug_idx,
             test_microbe_idx=test_microbe_idx,
             test_labels=test_labels)
    # 训练模型
    # 训练模型
    # model, decoder , node_gradients= train_gcn(
    #     train_data, A, drug_features, microbe_features,
    #     epochs=650, lr=0.005, hidden=256, dropout=0.2, device='cpu'
    # )
    fold_dir = os.path.join(base_path, f'fold{fold + 1}')
    os.makedirs(fold_dir, exist_ok=True)
    # model_path = os.path.join(fold_dir, f'{args.dataset}_gcn_model.pth')
    # decoder_path = os.path.join(fold_dir, f'{args.dataset}_decoder.pth')
    # model_path_fused = os.path.join(fold_dir, f'{args.dataset}_gcn_model_fused.pth')
    # decoder_path_fused = os.path.join(fold_dir, f'{args.dataset}_decoder_fused.pth')
    # # --- 建议修改 ---
    # 把关键参数 hidden_dim 加入文件名
    model_path = os.path.join(fold_dir, f'{args.dataset}_hd{args.hidden_dim}_gcn_model.pth')
    decoder_path = os.path.join(fold_dir, f'{args.dataset}_hd{args.hidden_dim}_decoder.pth')
    model_path_fused = os.path.join(fold_dir, f'{args.dataset}_hd{args.hidden_dim_retrain}_gcn_model_fused.pth')
    decoder_path_fused = os.path.join(fold_dir, f'{args.dataset}_hd{args.hidden_dim_retrain}_decoder_fused.pth')

    # 判断是否需要重新训练
    if not args.retrain and os.path.exists(model_path) and os.path.exists(decoder_path):
        print(f'Fold {fold + 1}: 检测到已保存的模型，直接加载模型和解码器')
        model = GCNWithMLP(
            drug_in_dim=drug_fg.shape[1],
            drug_out_dim=drug_fg.shape[0],
            microbe_dim=microbe_features.shape[1],
            microbe_out_dim=microbe_features.shape[1],  # 使用原始微生物维度
            gcn_hidden=args.hidden_dim,
            dropout=args.dropout,
            use_microbe_mlp=False,  # 只降维药物
            dataset_name = args.dataset
        )


        decoder = MLPDecoder(args.hidden_dim)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        model = model.to(device)
        decoder = decoder.to(device)
        node_gradients = np.zeros((drug_features.shape[0] + microbe_features.shape[0], args.hidden_dim))  # 如果后续用不到，可以不管

    else:
        print(f'Fold {fold + 1}: 开始训练模型')
        #A_fold = np.array(np.where(A_fold > 0))
        #A_fold = torch.tensor(A_fold, dtype=torch.long, device=device)

        # ... existing code ...
        # if args.dataset in ['MDAD', 'aBiofilm']:
        #     batch_size = 256  # 你可以根据显存调整
        #     model, decoder, node_gradients, embeddings, X = train_gcn(
        #         train_data, edge_index, edge_weight, drug_fg_norm, drug_features_norm, drug_bert_norm,
        #         microbe_features_norm, microbe_bert_norm, microbe_path_norm, microbe_offset,
        #         epochs=args.epochs, lr=args.learning_rate, hidden=args.hidden_dim, dropout=args.dropout, device=device,
        #         args=args, batch_size=batch_size
        #     )
        # else:
        #     # 不传 batch_size
        #     model, decoder, node_gradients, embeddings, X = train_gcn(
        #         train_data, edge_index, edge_weight, drug_fg_norm, drug_features_norm, drug_bert_norm,
        #         microbe_features_norm, microbe_bert_norm, microbe_path_norm, microbe_offset,
        #         epochs=args.epochs, lr=args.learning_rate, hidden=args.hidden_dim, dropout=args.dropout, device=device,
        #         args=args
        #     )
        # ... existing code ...




        # if args.dataset in ['MDAD', 'aBiofilm']:
        #     batch_size = 256  # 你可以根据显存调整
        #     model, decoder, node_gradients, embeddings, X = train_gcn(
        #         train_data, edge_index, edge_weight, drug_fg_norm, drug_features_norm, drug_bert_norm,
        #         microbe_features_norm, microbe_bert_norm, microbe_path_norm, microbe_offset,
        #         epochs=args.epochs, lr=args.learning_rate, hidden=args.hidden_dim, dropout=args.dropout, device=device,
        #         args=args, batch_size=batch_size,
        #         test_data=test_data  # <--- 在这里添加
        #     )
        # else:
        #     # 不传 batch_size
        #     model, decoder, node_gradients, embeddings, X = train_gcn(
        #         train_data, edge_index, edge_weight, drug_fg_norm, drug_features_norm, drug_bert_norm,
        #         microbe_features_norm, microbe_bert_norm, microbe_path_norm, microbe_offset,
        #         epochs=args.epochs, lr=args.learning_rate, hidden=args.hidden_dim, dropout=args.dropout, device=device,
        #         args=args,
        #         test_data=test_data  # <--- 在这里添加
        #     )
        # main.py, ~line 340
        if args.dataset in ['MDAD', 'aBiofilm']:
            batch_size = 256  # 你可以根据显存调整
            model, decoder, node_gradients, embeddings, X = train_gcn(
                train_data, edge_index, edge_weight, drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm, microbe_offset,
                epochs=args.epochs, lr=args.learning_rate, hidden=args.hidden_dim, dropout=args.dropout, device=device,
                args=args, batch_size=batch_size,
                #test_data=test_data,
                test_data=val_data,
                fold_num=fold,
                save_dir=base_path,  # <--- 修改为 base_path
                plot_filename=f'training_curve_fold{fold + 1}_initial.png',  # <--- 新增此行
                weight_decay = args.wd
            )
        else:

            # 不传 batch_size
            model, decoder, node_gradients, embeddings, X = train_gcn(
                train_data, edge_index, edge_weight, drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm, microbe_offset,
                epochs=args.epochs, lr=args.learning_rate, hidden=args.hidden_dim, dropout=args.dropout, device=device,
                args=args,
                #test_data=test_data,
                test_data=val_data,
                fold_num=fold,
                save_dir=base_path,  # <--- 修改为 base_path
                plot_filename=f'training_curve_fold{fold + 1}_initial.png' , # <--- 新增此行
                weight_decay = args.wd
            )

        #下面的2行代码是保存原有模型和解码器,加到re里面去了
        torch.save(model.state_dict(), model_path)
        torch.save(decoder.state_dict(), decoder_path)


    # # 2. 保存模型参数副本，，，，加到re里面去了
    # old_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    # old_params_decoder = {name: p.clone().detach() for name, p in decoder.named_parameters()}
    # # 3. 计算Fisher信息(增量学习)(先暂时注释掉)
    # # Fisher信息计算（兼容PyG输入），，，，，加到re里面去了
    # fisher, fisher_decoder = compute_fisher_gcn(
    #     model, decoder, train_data, edge_index, edge_weight,
    #     drug_fg_norm, drug_features_norm, drug_bert_norm,
    #     microbe_features_norm, microbe_bert_norm, microbe_path_norm,
    #     microbe_offset, device=device
    # )
    #
    #
    # # ========== 新增：保存Fisher信息和old_params ==========，，，，，，加到re里面去了
    # torch.save(fisher, os.path.join(fold_dir, 'fisher.pth'))
    # torch.save(fisher_decoder, os.path.join(fold_dir, 'fisher_decoder.pth'))
    # torch.save(old_params, os.path.join(fold_dir, 'old_params.pth'))
    # torch.save(old_params_decoder, os.path.join(fold_dir, 'old_params_decoder.pth'))

    print("正在将Numpy特征数据转换为PyTorch Tensors以进行评估...")
    drug_fg_norm = torch.tensor(drug_fg_norm, dtype=torch.float32, device=device)
    drug_features_norm = torch.tensor(drug_features_norm, dtype=torch.float32, device=device)
    drug_bert_norm = torch.tensor(drug_bert_norm, dtype=torch.float32, device=device)
    microbe_features_norm = torch.tensor(microbe_features_norm, dtype=torch.float32, device=device)
    microbe_bert_norm = torch.tensor(microbe_bert_norm, dtype=torch.float32, device=device)
    microbe_path_norm = torch.tensor(microbe_path_norm, dtype=torch.float32, device=device)

    # 评估模型
    # 在每折for循环内“评估模型”阶段
    # 新增如下：——
    # train_auc, train_aupr = evaluate_gcn(
    #     model, decoder, train_data, edge_index, edge_weight,
    #     drug_fg_norm, drug_features_norm, drug_bert_norm,
    #     microbe_features_norm, microbe_bert_norm, microbe_path_norm,
    #     microbe_offset, device=device
    # )
    # print(f'Fold {fold + 1} Train AUC: {train_auc:.4f}, Train AUPR: {train_aupr:.4f}')
    #
    train_auc, train_aupr, train_acc = evaluate_gcn(
        model, decoder, train_data, edge_index, edge_weight,
        drug_fg_norm, drug_features_norm, drug_bert_norm,
        microbe_features_norm, microbe_bert_norm, microbe_path_norm,
        microbe_offset, device=device
    )
    print(f'Fold {fold + 1} Train AUC: {train_auc:.4f}, AUPR: {train_aupr:.4f}, ACC: {train_acc:.4f}')

    # ——已有代码（测试集）——
    # auc, aupr = evaluate_gcn(
    #     model, decoder, test_data, edge_index, edge_weight,
    #     drug_fg_norm, drug_features_norm, drug_bert_norm,
    #     microbe_features_norm, microbe_bert_norm, microbe_path_norm,
    #     microbe_offset, device=device
    # )
    # print(f'Fold {fold + 1} Test  AUC: {auc:.4f}, Test  AUPR: {aupr:.4f}')
    #
    # results.append((auc, aupr))
    auc, aupr, acc = evaluate_gcn(
        model, decoder, test_data, edge_index, edge_weight,
        drug_fg_norm, drug_features_norm, drug_bert_norm,
        microbe_features_norm, microbe_bert_norm, microbe_path_norm,
        microbe_offset, device=device
    )
    print(f'Fold {fold + 1} Test  AUC: {auc:.4f}, AUPR: {aupr:.4f}, ACC: {acc:.4f}')

    results.append((auc, aupr, acc))







    # ================= GNNExplainer解释部分（加了详细调试信息） =================
    import torch
    import numpy as np
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    jieshi=1#决定是否用GNNExplainer解释
    camm=0#决定是否用CAM
    nojieshi=0#决定要不要用解释获得增强邻接矩阵，然后进行增量学习
    # ...前面模型训练和评估部分...


        # 1. 获取特征
    with torch.no_grad():
        X_input = (
            torch.tensor(drug_fg_norm, dtype=torch.float32, device=device),
            torch.tensor(drug_features_norm, dtype=torch.float32, device=device),
            torch.tensor(drug_bert_norm, dtype=torch.float32, device=device),
            torch.tensor(microbe_features_norm, dtype=torch.float32, device=device),
            torch.tensor(microbe_bert_norm, dtype=torch.float32, device=device),
            torch.tensor(microbe_path_norm, dtype=torch.float32, device=device),
        )
        embeddings, X = model(X_input, edge_index, edge_weight)
    import torch

    # --- 【核心修改】在解释循环开始前，强制释放所有非必需的显存 ---
    print("解释前，手动释放embeddings等大张量...")
    embeddings_cached = embeddings.clone().detach()

    del embeddings  # 删除不再需要的巨大张量
    import gc

    gc.collect()


    torch.cuda.empty_cache()

    if (nojieshi == 1):
        print(f"Fold {fold + 1}: 不使用解释器增强，直接将当前模型另存为 _fused 版本")

        # 1) 直接保存当前模型为 _fused
        torch.save(model.state_dict(), model_path_fused)
        torch.save(decoder.state_dict(), decoder_path_fused)
        print(f"已保存: {model_path_fused}")
        print(f"已保存: {decoder_path_fused}")

        # 2) 计算并保存 EWC 所需的 Fisher 信息与旧参数
        old_params = {name: p.clone().detach() for name, p in model.named_parameters()}
        old_params_decoder = {name: p.clone().detach() for name, p in decoder.named_parameters()}

        fisher, fisher_decoder = compute_fisher_gcn(
            model, decoder, train_data, edge_index, edge_weight,
            drug_fg_norm, drug_features_norm, drug_bert_norm,
            microbe_features_norm, microbe_bert_norm, microbe_path_norm,
            microbe_offset, device=device
        )

        torch.save(fisher, os.path.join(fold_dir, 'fisher.pth'))
        torch.save(fisher_decoder, os.path.join(fold_dir, 'fisher_decoder.pth'))
        torch.save(old_params, os.path.join(fold_dir, 'old_params.pth'))
        torch.save(old_params_decoder, os.path.join(fold_dir, 'old_params_decoder.pth'))
        print("已保存: fisher.pth, fisher_decoder.pth, old_params.pth, old_params_decoder.pth")

        # 3) 可选：评估一下并作为 fused 结果记录，便于后面统一打印
        auc_fused_direct, aupr_fused_direct,acc_fused_direct = evaluate_gcn(
            model, decoder, test_data, edge_index, edge_weight,
            drug_fg_norm, drug_features_norm, drug_bert_norm,
            microbe_features_norm, microbe_bert_norm, microbe_path_norm,
            microbe_offset, device=device
        )
        print(f'【直接保存为_fused】Fold {fold + 1} Test AUC: {auc_fused_direct:.4f}, AUPR: {aupr_fused_direct:.4f}')
        results_fused.append((auc_fused_direct, aupr_fused_direct,acc_fused_direct))

    if jieshi ==1 :#是否用GNNExplainer进行解释和得到增强邻接矩阵

#         num_total_nodes = X.shape[0]
#         #importance_matrix = np.zeros((num_total_nodes, num_total_nodes))
#         importance_matrix_sum = np.zeros((num_total_nodes, num_total_nodes))
#         bianshu=15#15
#         print(f'选取训练集前{bianshu}条边进行GNNExplainer')
#
#         # print(f'随机选取训练集中的【正样本】边 {bianshu} 条进行GNNExplainer')
#         #
#         # # 核心修正：必须从 train_pos (只包含正样本) 中进行随机抽样
#         # num_train_pos_samples = len(train_pos)
#         # # 确保采样数量不超过训练正样本总数
#         # num_to_sample = min(bianshu, num_train_pos_samples)
#         #
#         # # 从正样本中随机选择 bianshu 个不重复的索引
#         # random_indices = np.random.choice(num_train_pos_samples, num_to_sample, replace=False)
#         #
#         # # 遍历这些随机选出的正样本索引
#         # for ii in random_indices:
#         print(f'选取训练集中的【正样本】边 {bianshu} 条进行GNNExplainer')
#
#         num_to_select = min(bianshu, len(train_pos_new))
#         scored_edges = []
#         with torch.no_grad():
#             for drug_idx, microbe_idx in train_pos_new:
#                 src = drug_idx
#                 dst = microbe_idx + microbe_offset
#                 logits = decoder(
#                     embeddings_cached[src].unsqueeze(0),
#                     embeddings_cached[dst].unsqueeze(0)
#                 )
#                 prob = torch.sigmoid(logits).item()
#                 scored_edges.append((prob, (drug_idx, microbe_idx)))
#
#         top_edges = [edge for _, edge in sorted(scored_edges, reverse=True)[:num_to_select]]
#
#         for drug_idx, microbe_idx in top_edges:
#
#             # 2. 选择要解释的边，直接从 train_pos 中获取，确保是真实存在的边
#             # edge_to_explain = train_pos[ii]
#             # drug_idx = edge_to_explain[0]
#             # microbe_idx = edge_to_explain[1]
#
#
#             #调试信息
#             src = drug_idx
#             dst = microbe_idx + microbe_offset
#             edge_to_explain = torch.tensor([[src], [dst]], device=device)
#             # ...原来循环里的其余逻辑照常用 drug_idx / microbe_idx ...
#             print(f"==== 准备解释边 src={src} -> dst={dst}（microbe_offset={microbe_offset}）====")
#
#             # 打印全图size
#             print(f"全图节点数: {X.shape[0]}, 全图边数: {edge_index.shape[1]}")
#
#             if torch.cuda.is_available():
#                 print(
#                     f"[调试] GPU allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB, reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")
#
#
#
#             # ================== 核心修正点 1 ==================
#             # 在每次循环开始时，重置单次解释的重要性矩阵
#             importance_matrix = np.zeros((num_total_nodes, num_total_nodes))
#             # ================================================
#             src = drug_idx
#
#             dst = microbe_idx+microbe_offset
#             print(f"准备解释的边：药物节点 {src} -> 微生物节点 {dst-microbe_offset}")
#             #print(f"edge_index最大节点编号={edge_index.max()}, X.shape[0]={X.shape[0]}")
#             # 3. 初始化explainer
#             from torch_geometric.explain import Explainer
#             from torch_geometric.explain.algorithm import GNNExplainer
#             from torch_geometric.explain.config import ModelConfig, ModelReturnType
#
#             #wrapped_model = GCNWithDecoderWrapper(model, decoder, microbe_offset).to(device)
#             # ======================= 【核心修改处】 =======================
#             # 根据数据集选择不同的 Wrapper
#             if args.dataset == 'aBiofilm':
#                 print("信息: 检测到 aBiofilm 数据集，使用支持批处理的 GCNWithDecoderWrapperBatched。")
#                 wrapped_model = GCNWithDecoderWrapperBatched(
#                     gcn_model=model,
#                     decoder=decoder,
#                     microbe_offset=microbe_offset,
#                     batch_size=4096  # 这个值可以根据你的显存大小调整，如果还爆显存就调小
#                 ).to(device)
#             else:
#                 print("信息: 使用标准的 GCNWithDecoderWrapper。")
#                 wrapped_model = GCNWithDecoderWrapper(model, decoder, microbe_offset).to(device)
#             # =============================================================
#
#             wrapped_model.eval()
#
# #释放显存
#
#
#             explainer = Explainer(
#                 model=wrapped_model,
#                 algorithm=GNNExplainer(epochs=100,#100
#                                        # edge_ent=1.0,    # 提高掩码稀疏性
#                                        #  edge_size=0.05
#                                        ),# 控制保留比例),#自己设置GNNExplainer的训练轮数
#                 explanation_type='phenomenon',
#                 edge_mask_type='object',
#                 node_mask_type='attributes',
#                 model_config=ModelConfig(
#                     mode='binary_classification',
#                     task_level='edge',
#                     return_type=ModelReturnType.raw,
#                 ),
#             )
#
#             # 构造所有边的标签
#             row_np = edge_index[0].cpu().numpy()
#             col_np = edge_index[1].cpu().numpy()
#             labels = []
#             for src_, dst_ in zip(row_np, col_np):
#                 # 判断是否为药物-微生物边
#                 if src_ < microbe_offset and dst_ >= microbe_offset:
#                     label = int(I_train[src_, dst_ - microbe_offset] == 1)
#                 else:
#                     label = 0
#                 labels.append(label)
#             edge_labels = torch.tensor(labels, dtype=torch.long, device=X.device)
#
#             # === 根据数据集选择解释模式 ===
#             if args.dataset == 'aBiofilm':
#                 explain_mode = 'full'
#             else:
#                 explain_mode = 'full'
#
#             # 4. 选择解释方式
#
#             if explain_mode == 'full':
#                 # explanation, edge_index_used = explain_with_full_graph(
#                 #     explainer, X, edge_index, edge_weight, src, dst, edge_labels)
#
#                 #改成混合精度
#                 from torch.cuda.amp import autocast
#                 use_amp_explainer = (args.dataset in ['MDAD', 'aBiofilm']) and ('cuda' in str(device))
#                 with autocast(enabled=use_amp_explainer):
#                     explanation, edge_index_used = explain_with_full_graph(
#                         explainer, X, edge_index, edge_weight, src, dst, edge_labels
#                     )
#
#
#                 subset_used = None
#             else:
#                 explanation, edge_index_used, subset_used = explain_with_khop_subgraph(
#                     explainer, X, edge_index, edge_weight, src, dst, num_hops=2)
#
#             # 5. 输出结果
#             edge_mask = explanation.edge_mask
#             edge_mask_np = edge_mask.detach().cpu().numpy() if isinstance(edge_mask, torch.Tensor) else edge_mask
#             topk = 15#起到一个查看前20条重要的边的作用
#             top_indices = np.argsort(-edge_mask_np)[:topk]
#             f1=0
#             if(f1==1):
#                 print(f"\n【最重要的前{topk}条边】")
#                 for rank, idx in enumerate(top_indices, 1):
#                     src_node = edge_index_used[0, idx].item()
#                     dst_node = edge_index_used[1, idx].item()
#                     # if subset_used is not None:
#                     #     global_src = subset_used[src_node]
#                     #     global_dst = subset_used[dst_node]
#                     #     print(
#                     #         f"Top{rank}: (本地节点{src_node}->{dst_node}, 全局节点{global_src}->{global_dst}), 重要性={edge_mask_np[idx]:.4f}")
#                     # else:
#                     #     print(f"Top{rank}: (全局节点{src_node}->{dst_node}), 重要性={edge_mask_np[idx]:.4f}")
#                     if subset_used is not None:
#                         global_src = subset_used[src_node]
#                         global_dst = subset_used[dst_node]
#                     else:
#                         global_src = src_node
#                         global_dst = dst_node
#
#
#                     def node_label(node_id):
#                         if node_id < microbe_offset:
#                             return f"药物节点 {node_id}"
#                         else:
#                             return f"微生物节点 {node_id - microbe_offset}"
#
#
#                     print(
#                         f"Top{rank}: ({node_label(global_src)} -> {node_label(global_dst)}), 重要性={edge_mask_np[idx]:.4f}"
#                     )
#
#             # 2. 创建一个空的邻接矩阵来存储边的重要性分数
#             #importance_matrix = np.zeros((num_total_nodes, num_total_nodes))
#
#             # 3. 使用GNNExplainer计算出的边和重要性分数来填充矩阵
#             edge_src_nodes = edge_index_used[0].cpu().numpy()
#             edge_dst_nodes = edge_index_used[1].cpu().numpy()
#             for i in range(len(edge_mask_np)):
#                 u, v = edge_src_nodes[i], edge_dst_nodes[i]
#                 score = edge_mask_np[i]
#                 importance_matrix[u, v] = score
#                 importance_matrix[v, u] = score  # 对称
#             importance_matrix_sum += importance_matrix  # 累加
#
#             # --- 新增：每次循环后清理显存 ---
#             del explainer, wrapped_model, explanation, edge_mask
#             gc.collect()
#             torch.cuda.empty_cache()
#             # ---------------------------------
#
#
#
#         avg_importance_matrix = importance_matrix_sum / (bianshu)*0.1 # 求平均再乘一个自定义的系数
#         #增加去噪（比如保留全局前 1% 的高权重边）
#         # threshold = np.percentile(avg_importance_matrix[avg_importance_matrix > 0], 30)#30
#         # avg_importance_matrix[avg_importance_matrix < threshold] = 0
#         positive_mask = avg_importance_matrix > 0
#         if np.any(positive_mask):
#             threshold = np.percentile(avg_importance_matrix[positive_mask], 30)
#             avg_importance_matrix[avg_importance_matrix < threshold] = 0
#         else:
#             print("GNNExplainer 这一折的 edge_mask 全是 0，跳过阈值裁剪。")
# vvvvvvvvvvvvvvvv 【从这里开始替换】 vvvvvvvvvvvvvvvv

        # --- 步骤1: 定义重要性矩阵的缓存文件路径 ---
        importance_matrix_path = os.path.join(fold_dir, f'avg_importance_matrix_fold{fold + 1}.npy')

        # --- 步骤2: 检查开关和文件是否存在 ---
        if args.use and os.path.exists(importance_matrix_path):
            print(f"★★ 开关已启用，正在从缓存加载重要性矩阵 -> {importance_matrix_path}")
            avg_importance_matrix = np.load(importance_matrix_path)
            print("★★ 加载成功！跳过GNNExplainer计算。")
        else:
            # --- 如果不使用缓存或文件不存在，则执行原始的GNNExplainer计算 ---
            if args.use:
                print(f"★★ 开关已启用，但未找到缓存文件: {importance_matrix_path}。将重新计算。")

            print("★★ 开始执行GNNExplainer计算 (这可能需要一些时间)...")

            # 【这里是您原有的GNNExplainer计算循环，保持不变】
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            num_total_nodes = X.shape[0]
            importance_matrix_sum = np.zeros((num_total_nodes, num_total_nodes))
            bianshu = 2#15
            print(f'选取训练集前{bianshu}条边进行GNNExplainer')

            print(f'选取训练集中的【正样本】边 {bianshu} 条进行GNNExplainer')

            num_to_select = min(bianshu, len(train_pos_new))
            scored_edges = []
            with torch.no_grad():
                for drug_idx, microbe_idx in train_pos_new:
                    src = drug_idx
                    dst = microbe_idx + microbe_offset
                    logits = decoder(
                        embeddings_cached[src].unsqueeze(0),
                        embeddings_cached[dst].unsqueeze(0)
                    )
                    prob = torch.sigmoid(logits).item()
                    scored_edges.append((prob, (drug_idx, microbe_idx)))

            top_edges = [edge for _, edge in sorted(scored_edges, reverse=True)[:num_to_select]]

            for drug_idx, microbe_idx in top_edges:
                src = drug_idx
                dst = microbe_idx + microbe_offset
                edge_to_explain = torch.tensor([[src], [dst]], device=device)
                print(f"==== 准备解释边 src={src} -> dst={dst}（microbe_offset={microbe_offset}）====")
                print(f"全图节点数: {X.shape[0]}, 全图边数: {edge_index.shape[1]}")
                if torch.cuda.is_available():
                    print(
                        f"[调试] GPU allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB, reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

                importance_matrix = np.zeros((num_total_nodes, num_total_nodes))
                src = drug_idx
                dst = microbe_idx + microbe_offset
                print(f"准备解释的边：药物节点 {src} -> 微生物节点 {dst - microbe_offset}")

                from torch_geometric.explain import Explainer
                from torch_geometric.explain.algorithm import GNNExplainer
                from torch_geometric.explain.config import ModelConfig, ModelReturnType


                if args.dataset == 'a':
                    print("信息: 检测到 aBiofilm 数据集，使用支持批处理的 GCNWithDecoderWrapperBatched。")
                    wrapped_model = GCNWithDecoderWrapperBatched(
                        gcn_model=model,
                        decoder=decoder,
                        microbe_offset=microbe_offset,
                        batch_size=4096
                    ).to(device)
                else:
                    print("信息: 使用标准的 GCNWithDecoderWrapper。")
                    wrapped_model = GCNWithDecoderWrapper(model, decoder, microbe_offset).to(device)

                wrapped_model.eval()

                explainer = Explainer(
                    model=wrapped_model,
                    algorithm=GNNExplainer(epochs=100),
                    explanation_type='phenomenon',
                    edge_mask_type='object',
                    node_mask_type='attributes',
                    model_config=ModelConfig(
                        mode='binary_classification',
                        task_level='edge',
                        return_type=ModelReturnType.raw,
                    ),
                )

                row_np = edge_index[0].cpu().numpy()
                col_np = edge_index[1].cpu().numpy()
                labels = []
                for src_, dst_ in zip(row_np, col_np):
                    if src_ < microbe_offset and dst_ >= microbe_offset:
                        label = int(I_train[src_, dst_ - microbe_offset] == 1)
                    else:
                        label = 0
                    labels.append(label)
                edge_labels = torch.tensor(labels, dtype=torch.long, device=X.device)

                if args.dataset == 'aBiofilm':
                    explain_mode = 'full'
                else:
                    explain_mode = 'full'

                from torch.cuda.amp import autocast

                use_amp_explainer = (args.dataset in ['MDAD','aBiofilm']) and ('cuda' in str(device))#双精度训练
                with autocast(enabled=use_amp_explainer):
                    explanation, edge_index_used = explain_with_full_graph(
                        explainer, X, edge_index, edge_weight, src, dst, edge_labels
                    )
                subset_used = None

                edge_mask = explanation.edge_mask

                huatu=0#控制是否画热力图
                if(huatu==1):
                   #  # 【【【新增调用】】】调用新的热力图函数
                    visualize_as_heatmap(
                        edge_index_used=edge_index_used,
                        edge_mask=edge_mask,
                        target_drug_id=drug_idx,
                        target_microbe_id_local=microbe_idx,
                        microbe_offset=microbe_offset,
                        fold_dir=fold_dir,
                        fold_num=fold + 1  # 确保折数是从1开始
                    )

                    # --- 新增：每次循环后清理显存 ---
                   # del explainer, wrapped_model, explanation, edge_mask
                    gc.collect()
                    torch.cuda.empty_cache()
                   #  # ---------------------------------



                edge_mask_np = edge_mask.detach().cpu().numpy() if isinstance(edge_mask, torch.Tensor) else edge_mask

                topk = 15#起到一个查看前20条重要的边的作用
                top_indices = np.argsort(-edge_mask_np)[:topk]
                f1=1
                if(f1==1):
                    print(f"\n【最重要的前{topk}条边】")
                    for rank, idx in enumerate(top_indices, 1):
                        src_node = edge_index_used[0, idx].item()
                        dst_node = edge_index_used[1, idx].item()
                        # if subset_used is not None:
                        #     global_src = subset_used[src_node]
                        #     global_dst = subset_used[dst_node]
                        #     print(
                        #         f"Top{rank}: (本地节点{src_node}->{dst_node}, 全局节点{global_src}->{global_dst}), 重要性={edge_mask_np[idx]:.4f}")
                        # else:
                        #     print(f"Top{rank}: (全局节点{src_node}->{dst_node}), 重要性={edge_mask_np[idx]:.4f}")
                        if subset_used is not None:
                            global_src = subset_used[src_node]
                            global_dst = subset_used[dst_node]
                        else:
                            global_src = src_node
                            global_dst = dst_node


                        def node_label(node_id):
                            if node_id < microbe_offset:
                                return f"药物节点 {node_id}"
                            else:
                                return f"微生物节点 {node_id - microbe_offset}"


                        print(
                            f"Top{rank}: ({node_label(global_src)} -> {node_label(global_dst)}), 重要性={edge_mask_np[idx]:.4f}"
                        )



                edge_src_nodes = edge_index_used[0].cpu().numpy()
                edge_dst_nodes = edge_index_used[1].cpu().numpy()
                for i in range(len(edge_mask_np)):
                    u, v = edge_src_nodes[i], edge_dst_nodes[i]
                    score = edge_mask_np[i]
                    importance_matrix[u, v] = score
                    importance_matrix[v, u] = score
                importance_matrix_sum += importance_matrix

                del explainer, wrapped_model, explanation, edge_mask
                gc.collect()
                torch.cuda.empty_cache()

            avg_importance_matrix = importance_matrix_sum / (bianshu) * 0.1
            positive_mask = avg_importance_matrix > 0
            if np.any(positive_mask):
                threshold = np.percentile(avg_importance_matrix[positive_mask], 30)
                avg_importance_matrix[avg_importance_matrix < threshold] = 0
            else:
                print("GNNExplainer 这一折的 edge_mask 全是 0，跳过阈值裁剪。")

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # 【这里是您原有的GNNExplainer计算循环的结束】

            # --- 步骤3: 【核心新增】在计算完成后，保存结果 ---
            print(f"★★ 计算完成，正在将重要性矩阵保存到 -> {importance_matrix_path}")
            np.save(importance_matrix_path, avg_importance_matrix)
            print("★★ 保存成功！")

            # ^^^^^^^^^^^^^^^^^^ 【在这里结束替换】 ^^^^^^^^^^^^^^^^^^
            fangansan=1
            if fangansan==1:
                # `top_edges` 列表是在前面计算avg_importance_matrix时生成的，我们可以直接复用
                num_samples_for_agg = len(top_edges)

                if num_samples_for_agg > 0:
                    # 调用新文件中的函数，并传入所有必需的参数
                    run_aggregate_visualization(
                        model=model,
                        decoder=decoder,
                        train_pos_samples_to_explain=top_edges,  # 传入预选的高置信度样本
                        embeddings_cached=embeddings_cached,  # 传入预计算的节点嵌入
                        X=X,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        I_train=I_train,  # 传入用于训练的关联矩阵
                        microbe_offset=microbe_offset,
                        fold_dir=fold_dir,
                        fold_num=fold + 1,
                        args=args,
                        device=device
                    )
                else:
                    print("警告: [方案三] 未找到可供解释的样本，跳过聚合可视化。")
            # ========================================================================

#
        # #
        # #
        # #
        # # ==================== 【新增】对 avg_importance_matrix 进行归一化 ====================
        # print("对 avg_importance_matrix 进行归一化...")
        #
        # # 1. 找出所有非零重要性分数，以避免大量的0值影响归一化
        # non_zero_scores = avg_importance_matrix[avg_importance_matrix > 0]
        #
        # # 2. 检查是否存在非零分数，防止数组为空
        # if non_zero_scores.size > 0:
        #     max_score = non_zero_scores.max()
        #     min_score = non_zero_scores.min()
        #
        #     # 3. 检查最大值和最小值是否相等，防止除以零
        #     if max_score > min_score:
        #         # Min-Max Scaling 公式: (value - min) / (max - min)
        #         # 只对非零部分进行计算和赋值，保持矩阵的稀疏性
        #         avg_importance_matrix[avg_importance_matrix > 0] = (non_zero_scores - min_score) / (
        #                     max_score - min_score)
        #         print(
        #             f"归一化完成 (Min-Max)。分数范围已缩放到: [{avg_importance_matrix.min():.4f}, {avg_importance_matrix.max():.4f}]")
        #     else:
        #         # 如果所有非零值都相等，则将它们全部设置为1（最大重要性）
        #         avg_importance_matrix[avg_importance_matrix > 0] = 1.0
        #         print("所有非零重要性分数相同，已全部设置为1.0")
        # else:
        #     print("警告: avg_importance_matrix 中没有非零值，无需归一化。")
        # # ==============================================================================
        #
        # avg_importance_matrix=avg_importance_matrix*0.1



        # 用平均矩阵做后续操作
        A_fused = fuse_adj_with_mask(Sd, Sm, I_train, avg_importance_matrix)
            # 后面评估的代码不变
            # === 1. 融合邻接矩阵（直接用 importance_matrix 作为mask）===


        # === 2. 转成edge_index/edge_weight（和训练流程一致）===
        row_f, col_f = np.where(A_fused != 0)
        edge_index_f = np.stack([row_f, col_f], axis=0)
        edge_weight_f = A_fused[row_f, col_f]
        edge_index_f = torch.tensor(edge_index_f, dtype=torch.long, device=device)
        edge_weight_f = torch.tensor(edge_weight_f, dtype=torch.float32, device=device)

        # === 3. 直接评估（也可以重训练）===重新训练吧
        # 下面以 evaluate_gcn 为例，如果想train+eval就用train_gcn
        # auc_fused, aupr_fused = evaluate_gcn(
        #     model, decoder, test_data, edge_index_f, edge_weight_f,
        #     drug_fg_norm, drug_features_norm, drug_bert_norm,
        #     microbe_features_norm, microbe_bert_norm, microbe_path_norm,
        #     microbe_offset, device=device
        # )
        # print(f'【GNNExplainer增强邻接矩阵】Test AUC: {auc_fused:.4f}, AUPR: {aupr_fused:.4f}')
        # results_fused.append((auc_fused, aupr_fused))  # 新增


        re = 0 if (not args.retrain and os.path.exists(model_path_fused) and os.path.exists(decoder_path_fused)) else 1
        re=1#先暂时强制用增强邻接矩阵重新训练
        if (re == 1):  # 是否用增强邻接矩阵重新训练
            # ---- 重新训练前显存清理 ----
            try:
                del explainer, wrapped_model, explanation, edge_mask
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[显存清理] 准备进入 Fold {fold + 1} 的 GNNExplainer 重训练阶段，已释放显存")

            print(f"Fold {fold + 1}: 用增强邻接矩阵重新训练模型")
            # model_fused, decoder_fused, _, _, _ = train_gcn(
            #     train_data, edge_index_f, edge_weight_f,
            #     drug_fg_norm, drug_features_norm, drug_bert_norm,
            #     microbe_features_norm, microbe_bert_norm, microbe_path_norm,
            #     microbe_offset,
            #     epochs=args.epochs, lr=args.learning_rate, hidden=args.hidden_dim, dropout=args.dropout,
            #     device=device, args=args
            # )
            # main.py, ~line 615
            # main.py, ~line 705
            model_fused, decoder_fused, _, _, _ = train_gcn(
                train_data, edge_index_f, edge_weight_f,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset,
                epochs=args.epochs_retrain, lr=args.lr_retrain, hidden=args.hidden_dim_retrain, dropout=args.dropout_retrain,
                device=device, args=args,
                #test_data=test_data,
                test_data=val_data,
                fold_num=fold,
                save_dir=base_path,
                plot_filename=f'training_curve_fold{fold + 1}_fused.png',
                weight_decay = args.wd_retrain,
                # --- 【在这里修改】: 激活早停 ---
                use_early_stopping=True,
                patience=args.early_stopping_patience
                # --------------------------------


            )



            # torch.save(model.state_dict(), model_path_fused)
            # torch.save(decoder.state_dict(), decoder_path_fused)
            torch.save(model_fused.state_dict(), model_path_fused)
            torch.save(decoder_fused.state_dict(), decoder_path_fused)

            #新增保存fisher和参数
            # 2. 保存模型参数副本，，，，加到re里面去了
            old_params = {name: p.clone().detach() for name, p in model_fused.named_parameters()}
            old_params_decoder = {name: p.clone().detach() for name, p in decoder_fused.named_parameters()}
            # 3. 计算Fisher信息(增量学习)(先暂时注释掉)
            # Fisher信息计算（兼容PyG输入），，，，，加到re里面去了
            fisher, fisher_decoder = compute_fisher_gcn(
                model_fused, decoder_fused, train_data, edge_index_f, edge_weight_f,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset, device=device
            )


            # ========== 新增：保存Fisher信息和old_params ==========，，，，，，加到re里面去了
            torch.save(fisher, os.path.join(fold_dir, 'fisher.pth'))
            torch.save(fisher_decoder, os.path.join(fold_dir, 'fisher_decoder.pth'))
            torch.save(old_params, os.path.join(fold_dir, 'old_params.pth'))
            torch.save(old_params_decoder, os.path.join(fold_dir, 'old_params_decoder.pth'))






            # auc_fused_re, aupr_fused_re = evaluate_gcn(
            #     model_fused, decoder_fused, test_data, edge_index_f, edge_weight_f,
            #     drug_fg_norm, drug_features_norm, drug_bert_norm,
            #     microbe_features_norm, microbe_bert_norm, microbe_path_norm,
            #     microbe_offset, device=device
            # )
            # print(f'【增强邻接矩阵重新训练】Fold {fold + 1} Test AUC: {auc_fused_re:.4f}, AUPR: {aupr_fused_re:.4f}')
            # results_fused.append((auc_fused_re, aupr_fused_re))
            # ... (这里是保存 fisher 和 old_params 的代码) ...

            # --- 【核心修改】评估增强后的模型在训练集上的表现，并根据开关导出预测分数 ---
            save_fused_train_predictions_to_excel = True  # 开关：设为True以导出Excel

            if save_fused_train_predictions_to_excel:
                # 调用函数并请求返回概率值
                # 注意：这里使用增强后的图结构 edge_index_f, edge_weight_f
                train_auc_fused, train_aupr_fused, train_probs_fused = evaluate_gcn(
                    model_fused, decoder_fused, train_data, edge_index_f, edge_weight_f,
                    drug_fg_norm, drug_features_norm, drug_bert_norm,
                    microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                    microbe_offset, device=device,
                    return_probs=True  # 关键：让函数返回概率
                )
                print(
                    f'【增强后模型-训练集】Fold {fold + 1} Train AUC: {train_auc_fused:.4f}, Train AUPR: {train_aupr_fused:.4f}')

                # --- 保存到Excel文件的逻辑 ---
                train_drug_idx, train_microbe_idx, _ = train_data
                predictions_df = pd.DataFrame({
                    'Drug_ID': train_drug_idx,
                    'Microbe_ID': train_microbe_idx,
                    'Predicted_Probability': train_probs_fused
                })

                # 文件名要区分开，表示这是增强后的结果
                output_filename = f'fused_train_set_predictions_fold_{fold + 1}.xlsx'
                output_path = os.path.join(fold_dir, output_filename)

                predictions_df.to_excel(output_path, index=False)
                print(f"成功: 增强后训练集预测分数已保存至 -> {output_path}")

            else:
                # 如果不导出，只做简单评估
                train_auc_fused, train_aupr_fused = evaluate_gcn(
                    model_fused, decoder_fused, train_data, edge_index_f, edge_weight_f,
                    drug_fg_norm, drug_features_norm, drug_bert_norm,
                    microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                    microbe_offset, device=device
                )
                print(
                    f'【增强后模型-训练集】Fold {fold + 1} Train AUC: {train_auc_fused:.4f}, Train AUPR: {train_aupr_fused:.4f}')

            # --- 【原有代码】评估增强后的模型在测试集上的表现（保持不变） ---
            # auc_fused_re, aupr_fused_re = evaluate_gcn(
            #     model_fused, decoder_fused, test_data, edge_index_f, edge_weight_f,
            #     drug_fg_norm, drug_features_norm, drug_bert_norm,
            #     microbe_features_norm, microbe_bert_norm, microbe_path_norm,
            #     microbe_offset, device=device
            # )
            # print(f'【增强邻接矩阵重新训练】Fold {fold + 1} Test AUC: {auc_fused_re:.4f}, AUPR: {aupr_fused_re:.4f}')
            # results_fused.append((auc_fused_re, aupr_fused_re))

            auc_fused_re, aupr_fused_re, acc_fused_re = evaluate_gcn(
                model_fused, decoder_fused, test_data, edge_index_f, edge_weight_f,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset, device=device
            )
            print(
                f'【增强邻接矩阵重新训练】Fold {fold + 1} Test AUC: {auc_fused_re:.4f}, AUPR: {aupr_fused_re:.4f}, ACC: {acc_fused_re:.4f}')
            results_fused.append((auc_fused_re, aupr_fused_re, acc_fused_re))






            # === 每折结束后，彻底释放显存防止碎片化 ===(防止显存爆了)
            try:
                del model, decoder
                del model_fused, decoder_fused
            except NameError:
                pass
            del train_data, test_data
            del edge_index, edge_weight
            del drug_fg_norm, drug_features_norm, drug_bert_norm
            del microbe_features_norm, microbe_bert_norm, microbe_path_norm
            gc.collect()
            torch.cuda.empty_cache()

            # 可选：重置PyTorch显存统计
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()








        if re == 0:  # 如果不重新训练，直接评估已保存的增强邻接矩阵训练后的模型
            print(f"Fold {fold + 1}: 已加载增强邻接矩阵训练的模型，开始评估...")
            model_fused = GCNWithMLP(
                drug_in_dim=drug_fg.shape[1],
                drug_out_dim=drug_fg.shape[0],
                microbe_dim=microbe_features.shape[1],
                microbe_out_dim=microbe_features.shape[1],
                gcn_hidden=args.hidden_dim,
                dropout=args.dropout,
                use_microbe_mlp=False,  # 只降维药物
                dataset_name=args.dataset
            )
            decoder_fused = MLPDecoder(args.hidden_dim)

            model_fused.load_state_dict(torch.load(model_path_fused, map_location=device))
            decoder_fused.load_state_dict(torch.load(decoder_path_fused, map_location=device))
            model_fused = model_fused.to(device)
            decoder_fused = decoder_fused.to(device)

            auc_fused, aupr_fused = evaluate_gcn(
                model_fused, decoder_fused, test_data, edge_index_f, edge_weight_f,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset, device=device
            )
            print(f'【增强邻接矩阵训练后的模型】Fold {fold + 1} Test AUC: {auc_fused:.4f}, AUPR: {aupr_fused:.4f}')
            results_fused.append((auc_fused, aupr_fused))

            # ===== 每一折结束后，强制显存清理防止碎片化 =====
            vars_to_del = [
                'model', 'decoder', 'model_fused', 'decoder_fused',
                'explainer', 'wrapped_model', 'explanation', 'edge_mask',
                'train_data', 'test_data',
                'edge_index', 'edge_weight', 'edge_index_f', 'edge_weight_f',
                'drug_fg_norm', 'drug_features_norm', 'drug_bert_norm',
                'microbe_features_norm', 'microbe_bert_norm', 'microbe_path_norm',
                'importance_matrix', 'importance_matrix_sum', 'X', 'embeddings'
            ]
            for var in vars_to_del:
                if var in locals():
                    del locals()[var]

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()

            print(f"[Fold {fold + 1} 清理完成] 显存占用: "
                  f"{torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, "
                  f"预留: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")




    if (camm == 1):
        print("\n===== Grad-CAM: 开始聚合节点重要性以增强邻接矩阵 =====")

        num_total_nodes = X.shape[0]
        importance_matrix_sum = np.zeros((num_total_nodes, num_total_nodes))
        bianshu = 8  # 决定了解释多少条边来聚合重要性

        print(f'随机选取训练集中的【正样本】边 {bianshu} 条进行Grad-CAM解释')

        # 从 train_pos (只包含正样本) 中进行随机抽样
        num_train_pos_samples = len(train_pos)
        num_to_sample = min(bianshu, num_train_pos_samples)
        random_indices = np.random.choice(num_train_pos_samples, num_to_sample, replace=False)

        # 遍历这些随机选出的正样本索引
        for ii in random_indices:
            # 1. 选择要解释的边
            edge_to_explain = train_pos[ii]
            drug_idx = edge_to_explain[0]
            microbe_idx = edge_to_explain[1]

            src = drug_idx
            dst = microbe_idx + microbe_offset
            print(f"准备解释的边：药物节点 {src} -> 微生物节点 {dst - microbe_offset}")

            # 2. 初始化 Explainer
            from torch_geometric.explain import Explainer
            from torch_geometric.explain.config import ModelConfig, ModelReturnType

            wrapped_model = GCNWithDecoderWrapper_cam(model, decoder, microbe_offset).to(device)
            wrapped_model.eval()

            # --- 这是使用 CustomGradCAM 的核心部分 ---
            explainer = None  # 先初始化为None
            try:
                # ========================= 核心修正点 =========================
                # 修正了目标层的访问路径
                target_gcn_layer = wrapped_model.gcn.gcn.conv2
                # ============================================================

                print(f"成功定位到Grad-CAM的目标层: {target_gcn_layer}")

                explainer = Explainer(
                    model=wrapped_model,
                    algorithm=CustomGradCAM(target_layer=target_gcn_layer),
                    explanation_type='model',  # GradCAM生成的是节点重要性，所以用'model'
                    node_mask_type='attributes',  # 我们得到的是节点属性的掩码
                    model_config=ModelConfig(
                        mode='binary_classification',
                        task_level='edge',
                        return_type=ModelReturnType.raw,
                    ),
                )
            except AttributeError:
                print("错误：无法在模型中找到目标层。请再次检查模型结构。")
                # 如果找不到目标层，就跳过本次解释
                continue  # 跳到下一个循环

            # 3. 根据数据集选择解释模式 (k-hop or full graph)
            if args.dataset == 'aBiofilm':
                explain_mode = 'khop'
            else:
                explain_mode = 'full'

            # 4. 生成解释
            if explain_mode == 'full':
                explanation = explainer(X, edge_index, edge_weight=edge_weight, index=(src, dst))
                edge_index_used = edge_index
                subset_used = None
            else:  # k-hop subgraph 逻辑
                from torch_geometric.utils import k_hop_subgraph

                num_hops = 2
                subset_used, sub_edge_index, mapping, _ = k_hop_subgraph(
                    node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index, relabel_nodes=True)

                src_local = mapping[mapping == src].item()
                dst_local = mapping[mapping == dst].item()

                sub_x = X[subset_used]
                sub_edge_weight = None  # k-hop 不方便传递权重，简化处理

                explanation = explainer(sub_x, sub_edge_index, edge_weight=sub_edge_weight,
                                        index=(src_local, dst_local))
                edge_index_used = sub_edge_index

            # 5. 处理Grad-CAM的输出 (node_mask) 并转换为 edge_mask
            if explanation.node_mask is None:
                print(f"警告: 边 {src}->{dst} 的解释未能生成node_mask，跳过。")
                continue

            print("正在处理 Grad-CAM 的节点重要性 (node_mask)...")
            #node_mask = explanation.node_mask.cpu().numpy()
            node_mask = explanation.node_mask.detach().cpu().squeeze(-1).numpy()


            # 将节点重要性转换为边重要性：一条边的重要性是其两端节点重要性的均值
            edge_src_nodes = edge_index_used[0].cpu().numpy()
            edge_dst_nodes = edge_index_used[1].cpu().numpy()

            edge_mask_np = (node_mask[edge_src_nodes] + node_mask[edge_dst_nodes]) / 2.0
            print("已将节点重要性转换为边重要性。")

            # 6. 填充单次解释的重要性矩阵
            importance_matrix = np.zeros((num_total_nodes, num_total_nodes))

            # 如果是k-hop, 需要将局部节点映射回全局节点
            if subset_used is not None:
                global_edge_src = subset_used[edge_src_nodes].cpu().numpy()
                global_edge_dst = subset_used[edge_dst_nodes].cpu().numpy()
            else:
                global_edge_src = edge_src_nodes
                global_edge_dst = edge_dst_nodes

            for i in range(len(edge_mask_np)):
                u, v = global_edge_src[i], global_edge_dst[i]
                score = edge_mask_np[i]
                importance_matrix[u, v] = score
                importance_matrix[v, u] = score  # 对称填充

            importance_matrix_sum += importance_matrix  # 累加到总矩阵


        # --- 循环结束后，聚合所有解释结果 ---
        avg_importance_matrix = importance_matrix_sum / num_to_sample  # 求平均

        # 去噪：保留全局前70% percentile的非零高权重边
        if np.any(avg_importance_matrix > 0):
            threshold = np.percentile(avg_importance_matrix[avg_importance_matrix > 0], 70)
            avg_importance_matrix[avg_importance_matrix < threshold] = 0
            print(f"已应用去噪阈值: {threshold:.4f}")

        # === 1. 融合邻接矩阵（使用平均重要性矩阵作为mask）===
        A_fused = fuse_adj_with_mask(Sd, Sm, I_train, avg_importance_matrix)

        # === 2. 转成edge_index/edge_weight（和训练流程一致）===
        row_f, col_f = np.where(A_fused != 0)
        edge_index_f = np.stack([row_f, col_f], axis=0)
        edge_weight_f = A_fused[row_f, col_f]
        edge_index_f = torch.tensor(edge_index_f, dtype=torch.long, device=device)
        edge_weight_f = torch.tensor(edge_weight_f, dtype=torch.float32, device=device)

        # === 3. 使用增强邻接矩阵重新训练或评估 ===
        re = 0 if (not args.retrain and os.path.exists(model_path_fused) and os.path.exists(decoder_path_fused)) else 1

        if (re == 1):
            print(f"Fold {fold + 1}: 用Grad-CAM增强邻接矩阵重新训练模型")
            # main.py, ~line 903
            model_fused, decoder_fused, _, _, _ = train_gcn(
                train_data, edge_index_f, edge_weight_f,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset,
                epochs=args.epochs_retrain, lr=args.lr_retrain, hidden=args.hidden_dim_retrain, dropout=args.dropout_retrain,
                device=device, args=args
                # 如果你也想在这里画图，可以像上面一样添加 test_data, fold_num, save_dir, plot_filename 参数
            )

            torch.save(model_fused.state_dict(), model_path_fused)
            torch.save(decoder_fused.state_dict(), decoder_path_fused)

            # 保存 Fisher 信息和旧参数
            old_params = {name: p.clone().detach() for name, p in model_fused.named_parameters()}
            old_params_decoder = {name: p.clone().detach() for name, p in decoder_fused.named_parameters()}
            fisher, fisher_decoder = compute_fisher_gcn(
                model_fused, decoder_fused, train_data, edge_index_f, edge_weight_f,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset, device=device
            )
            torch.save(fisher, os.path.join(fold_dir, 'fisher.pth'))
            torch.save(fisher_decoder, os.path.join(fold_dir, 'fisher_decoder.pth'))
            torch.save(old_params, os.path.join(fold_dir, 'old_params.pth'))
            torch.save(old_params_decoder, os.path.join(fold_dir, 'old_params_decoder.pth'))

            auc_fused_re, aupr_fused_re ,acc_fused_re= evaluate_gcn(
                model_fused, decoder_fused, test_data, edge_index_f, edge_weight_f,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset, device=device
            )
            print(
                f'【Grad-CAM增强邻接矩阵重新训练】Fold {fold + 1} Test AUC: {auc_fused_re:.4f}, AUPR: {aupr_fused_re:.4f}')
            results_fused.append((auc_fused_re, aupr_fused_re,acc_fused_re))

        if re == 0:
            print(f"Fold {fold + 1}: 已加载Grad-CAM增强邻接矩阵训练的模型，开始评估...")
            model_fused = GCNWithMLP(
                drug_in_dim=drug_fg.shape[1], drug_out_dim=drug_fg.shape[0],
                microbe_dim=microbe_features.shape[1], microbe_out_dim=microbe_features.shape[1],
                gcn_hidden=args.hidden_dim_retrain, dropout=args.dropout, use_microbe_mlp=False,
                dataset_name=args.dataset
            )
            decoder_fused = MLPDecoder(args.hidden_dim)
            model_fused.load_state_dict(torch.load(model_path_fused, map_location=device))
            decoder_fused.load_state_dict(torch.load(decoder_path_fused, map_location=device))
            model_fused = model_fused.to(device)
            decoder_fused = decoder_fused.to(device)

            auc_fused, aupr_fused,acc_fused = evaluate_gcn(
                model_fused, decoder_fused, test_data, edge_index_f, edge_weight_f,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset, device=device
            )
            print(
                f'【Grad-CAM增强邻接矩阵训练后的模型】Fold {fold + 1} Test AUC: {auc_fused:.4f}, AUPR: {aupr_fused:.4f}')
            results_fused.append((auc_fused, aupr_fused,acc_fused ))

    if(jieshi==2):
        print("\n===== PGExplainer: 开始聚合边重要性以增强邻接矩阵 =====")

        # 1. 调用修正后的函数获取增强的I矩阵
        # 注意：这里传入 I_train 是为了避免使用测试集信息来训练Explainer，防止数据泄漏
        I_fused_pg, pg_edge_scores = explain_with_pgexplainer(
            model, decoder, X, edge_index, edge_weight, Sd, Sm, I_train, microbe_offset, args,
            device,
            n_sample_edges=500,  # 可调：用于聚合的样本边数量
            top_percent=0.5,  # 可调：筛选前0.5%的新边
            lambda_new=0.1  # 可调：新边的权重
        )

        # 2. 使用增强后的 I_fused_pg 构建新的全邻接矩阵 A_fused_pg
        A_fused_pg = build_gcn_adj(Sd, Sm, I_fused_pg)

        # 3. 将稠密矩阵 A_fused_pg 转换为 PyG 的 edge_index 和 edge_weight 格式
        row_f_pg, col_f_pg = np.where(A_fused_pg > 0)
        edge_index_f_pg = np.stack([row_f_pg, col_f_pg], axis=0)
        edge_weight_f_pg = A_fused_pg[row_f_pg, col_f_pg]
        edge_index_f_pg = torch.tensor(edge_index_f_pg, dtype=torch.long, device=device)
        edge_weight_f_pg = torch.tensor(edge_weight_f_pg, dtype=torch.float32, device=device)
        print(f"原始图有 {edge_index.shape[1]} 条边, PGExplainer增强后有 {edge_index_f_pg.shape[1]} 条边。")

        # 4. 检查是否需要重新训练，或直接加载已有的增强模型
        re_pg = 0 if (
                    not args.retrain and os.path.exists(model_path_fused) and os.path.exists(decoder_path_fused)) else 1

        if (re_pg == 1):
            print(f"Fold {fold + 1}: 使用 PGExplainer 增强的邻接矩阵重新训练模型...")
            model_fused, decoder_fused, _, _, _ = train_gcn(
                train_data, edge_index_f_pg, edge_weight_f_pg,
                drug_fg_norm, drug_features_norm, drug_bert_norm,
                microbe_features_norm, microbe_bert_norm, microbe_path_norm,
                microbe_offset,
                epochs=args.epochs, lr=args.learning_rate, hidden=args.hidden_dim, dropout=args.dropout,
                device=device, args=args
            )
            torch.save(model_fused.state_dict(), model_path_fused)
            torch.save(decoder_fused.state_dict(), decoder_path_fused)
        else:
            print(f"Fold {fold + 1}: 加载已有的、使用增强邻接矩阵训练的模型...")
            model_fused = GCNWithMLP(
                drug_in_dim=drug_fg.shape[1], drug_out_dim=drug_fg.shape[0],
                microbe_dim=microbe_features.shape[1], microbe_out_dim=microbe_features.shape[1],
                gcn_hidden=args.hidden_dim, dropout=args.dropout, use_microbe_mlp=False,
                dataset_name=args.dataset
            )
            decoder_fused = MLPDecoder(args.hidden_dim)
            model_fused.load_state_dict(torch.load(model_path_fused, map_location=device))
            decoder_fused.load_state_dict(torch.load(decoder_path_fused, map_location=device))
            model_fused = model_fused.to(device)
            decoder_fused = decoder_fused.to(device)

        # 5. 评估增强后的模型性能
        auc_fused, aupr_fused = evaluate_gcn(
            model_fused, decoder_fused, test_data, edge_index_f_pg, edge_weight_f_pg,
            drug_fg_norm, drug_features_norm, drug_bert_norm,
            microbe_features_norm, microbe_bert_norm, microbe_path_norm,
            microbe_offset, device=device
        )
        print(f'【PGExplainer增强邻接重训练】Fold {fold + 1} Test AUC: {auc_fused:.4f}, AUPR: {aupr_fused:.4f}')
        results_fused.append((auc_fused, aupr_fused))







    if(args.dataset=='DugVirus'):#目前只对这个数据集画图，因为其他的数据集数据量太大了
        # ================== 新增：邻接矩阵式热力图可视化 (已修正) ==================
        print("\n===== 开始绘制邻接矩阵式边重要性热力图... =====")
        import numpy as np
        import matplotlib.pyplot as plt

        # 解决matplotlib中文显示问题
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
        except Exception as e:
            print(f"注意：未能设置中文字体，图形中的中文可能显示为方块。错误: {e}")
            print("您可以尝试安装 'SimHei' 字体或替换为系统已有的中文字体，如 'Microsoft YaHei'。")

        # 1. 确定图的节点总数，以创建正确大小的矩阵
        # (!!! 修正点 !!!) 使用在当前作用域内的变量 X 来获取节点总数, 而不是 data.x
        num_total_nodes = X.shape[0]

        # 2. 创建一个空的邻接矩阵来存储边的重要性分数
        importance_matrix = np.zeros((num_total_nodes, num_total_nodes))

        # 3. 使用GNNExplainer计算出的边和重要性分数来填充矩阵
        # edge_index_used 包含了边的源节点和目标节点
        # edge_mask_np 包含了对应边的重要性分数
        edge_src_nodes = edge_index_used[0].cpu().numpy()
        edge_dst_nodes = edge_index_used[1].cpu().numpy()

        for i in range(len(edge_mask_np)):
            u, v = edge_src_nodes[i], edge_dst_nodes[i]
            score = edge_mask_np[i]
            importance_matrix[u, v] = score
            importance_matrix[v, u] = score  # GNNExplainer通常处理无向图，所以对称填充

        # 4. 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 10))

        # 使用 'hot' colormap，它从黑到红到黄再到白，符合热力图的直觉
        # 为了让没有边的位置（值为0）显示为白色背景，而不是黑色
        # 我们创建一个Masked Array，将值为0的位置标记出来
        masked_matrix = np.ma.masked_where(importance_matrix == 0, importance_matrix)

        # 获取 'hot' 颜色映射表
        cmap = plt.get_cmap('hot')
        # 将被标记（masked）的值（也就是值为0的边）的颜色设置为白色
        cmap.set_bad(color='white')
        cmap = plt.get_cmap('hot').copy()
        cmap.set_bad(color='#D3D3D3')  # 灰色代表无边/无重要性

        # 使用 imshow 绘制矩阵
        im = ax.imshow(masked_matrix, cmap=cmap, interpolation='nearest')

        # 5. 添加颜色条 (Colorbar)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Edge Importance (边重要性)', rotation=270, labelpad=20)

        # 6. 设置标题和坐标轴标签
        ax.set_title(f'GNNExplainer Edge Importance Heatmap (Fold {fold + 1})', fontsize=16)
        ax.set_xlabel('节点编号 (Node ID)')
        ax.set_ylabel('节点编号 (Node ID)')

        # 确保Y轴的原点在上方，这对于矩阵可视化是标准的
        ax.invert_yaxis()

        # 7. 保存图像
        save_path = os.path.join(fold_dir, f'importance_heatmap_fold{fold + 1}_edge_{src}_{dst}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"邻接矩阵热力图已保存到: {save_path}")
        plt.close(fig)  # 关闭图形，防止在循环中内存泄漏

        # ================== 可视化代码结束 ==================


    ff=0
    if(ff==1):#控制是否画节点梯度
        # ================== 节点梯度可视化（全局loss贡献版） ==================
        print("\n===== 计算并绘制【全局loss节点梯度】可视化 =====")
        model.eval()
        decoder.eval()

        # 1. 获取所有节点的嵌入
        embeddings_for_grad = embeddings.detach().clone().requires_grad_(True)

        # 2. 构造训练集所有正负样本的边
        # train_drug_idx, train_microbe_idx, train_labels 已在上文
        src_nodes = train_drug_idx
        dst_nodes = train_microbe_idx + microbe_offset
        labels = torch.tensor(train_labels, dtype=torch.float32, device=device)

        # 3. 取对应节点的嵌入
        x_src = embeddings_for_grad[src_nodes]
        x_dst = embeddings_for_grad[dst_nodes]

        # 4. 计算所有训练样本的预测分数
        pred_scores = decoder(x_src, x_dst).squeeze()  # shape: [num_samples]

        # 5. 损失函数（如二分类交叉熵/BCELoss）
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(pred_scores, labels)

        # 6. 清空旧梯度
        model.zero_grad()
        decoder.zero_grad()

        # 7. 反向传播
        loss.backward()

        # 8. 获取节点嵌入的梯度
        if embeddings_for_grad.grad is not None:
            node_gradients = embeddings_for_grad.grad
            grad_magnitudes = torch.norm(node_gradients, p=2, dim=1).cpu().numpy()
            # 9. 可视化Top-K节点
            top_k_nodes = 50
            if len(grad_magnitudes) < top_k_nodes:
                top_k_nodes = len(grad_magnitudes)
            top_indices = np.argsort(-grad_magnitudes)[:top_k_nodes]
            top_magnitudes = grad_magnitudes[top_indices]
            # 准备标签
            labels_list = []
            for node_idx in top_indices:
                if node_idx < microbe_offset:
                    labels_list.append(f"Drug {node_idx}")
                else:
                    labels_list.append(f"Microbe {node_idx - microbe_offset}")
            # 使用渐变色
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, top_k_nodes))
            # 绘图
            fig, ax = plt.subplots(figsize=(12, 15))
            ax.barh(np.arange(top_k_nodes), top_magnitudes[::-1], color=colors[::-1])
            ax.set_yticks(np.arange(top_k_nodes))
            ax.set_yticklabels(labels_list[::-1], fontsize=10)
            ax.set_xlabel('梯度L2范数（需要调整的幅度）', fontsize=12)
            ax.set_ylabel('节点', fontsize=12)
            ax.set_title(f'Top {top_k_nodes} 节点梯度大小（损失贡献）可视化', fontsize=16)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            fig.tight_layout()
            save_path_grad = os.path.join(fold_dir, f'global_node_gradient_top{top_k_nodes}_fold{fold + 1}.png')
            plt.savefig(save_path_grad, dpi=300)
            print(f"【全局loss节点梯度图】已保存到: {save_path_grad}")
            plt.close(fig)
        else:
            print("警告：未能计算出节点梯度 (embeddings_for_grad.grad is None)，跳过梯度图绘制。")

# aucs, auprs = zip(*results)
# print(f'5折平均AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
# print(f'5折平均AUPR: {np.mean(auprs):.4f} ± {np.std(auprs):.4f}')
# if results_fused:
#     aucs_fused, auprs_fused = zip(*results_fused)
#     print(f'【增强邻接矩阵重新训练】5折平均AUC: {np.mean(aucs_fused):.4f} ± {np.std(aucs_fused):.4f}')
#     print(f'【增强邻接矩阵重新训练】5折平均AUPR: {np.mean(auprs_fused):.4f} ± {np.std(auprs_fused):.4f}')
aucs, auprs, accs = zip(*results)
print(f'5折平均AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
print(f'5折平均AUPR: {np.mean(auprs):.4f} ± {np.std(auprs):.4f}')
print(f'5折平均ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}') # 新增
if results_fused:
    aucs_fused, auprs_fused, accs_fused = zip(*results_fused)
    print(f'【增强邻接矩阵重新训练】5折平均AUC: {np.mean(aucs_fused):.4f} ± {np.std(aucs_fused):.4f}')
    print(f'【增强邻接矩阵重新训练】5折平均AUPR: {np.mean(auprs_fused):.4f} ± {np.std(auprs_fused):.4f}')
    print(f'【增强邻接矩阵重新训练】5折平均ACC: {np.mean(accs_fused):.4f} ± {np.std(accs_fused):.4f}') # 新增

else:
    print("没有融合矩阵增强的AUC/AUPR结果，可能没开启解释流程。")






# ================== 增量学习：用DrugVirus微调MDAD模型 ==================
zengliang=1#决定时候进行增量学习
# ================== 增量学习（动态选择数据集） ==================
# 使用新的命令行参数来控制是否以及如何进行增量学习
if args.incremental_dataset is not None and args.incremental_dataset != args.dataset and zengliang==1:

    # --- 步骤 1: 根据选择动态设置新任务（增量学习任务）的变量 ---
    inc_dataset_name = args.incremental_dataset
    inc_base_path = f'./{inc_dataset_name}/'

    if inc_dataset_name == 'MDAD':
        inc_microbe_offset = 1373
    elif inc_dataset_name == 'DrugVirus':
        inc_microbe_offset = 175
    elif inc_dataset_name == 'aBiofilm':
        inc_microbe_offset = 1720
    else:
        raise ValueError(f"未知的增量学习数据集: {inc_dataset_name}")

    print(f"\n==== 增量学习：用 {inc_dataset_name} (新任务) 微调 {args.dataset} (旧任务) 模型 ====")

    # --- 步骤 2: 加载新旧两个任务的原始特征 ---
    print(f"步骤 1/5: 加载 {args.dataset} (旧) 和 {inc_dataset_name} (新) 的原始特征...")

    # 加载新任务特征
    inc_drug_features, inc_drug_bert, inc_drug_fg, \
        inc_microbe_features, inc_microbe_bert, inc_microbe_path = load_features(
        inc_base_path + 'drugfeatures.txt', inc_base_path + 'drug_bert.xlsx',
        inc_base_path + 'fingerprint.xlsx', inc_base_path + 'microbefeatures.txt',
        inc_base_path + 'microbe_bert.xlsx', inc_base_path + 'microbe_path.xlsx'
    )

    # 旧任务特征在主流程中已经加载好了，直接使用
    old_drug_features, old_drug_bert, old_drug_fg, \
        old_microbe_features, old_microbe_bert, old_microbe_path = \
        drug_features, drug_bert, drug_fg, microbe_features, microbe_bert, microbe_path

    # 将新旧特征打包成列表，方便后续处理
    inc_feats_raw = [
        inc_drug_fg, inc_drug_features, inc_drug_bert,
        inc_microbe_features, inc_microbe_bert, inc_microbe_path
    ]
    old_feats = [
        old_drug_fg, old_drug_features, old_drug_bert,
        old_microbe_features, old_microbe_bert, old_microbe_path
    ]

    # --- 步骤 3: 【外部对齐】预训练MLP并进行一次性变换 ---
    print("步骤 2/5: 执行【外部对齐】...")
    align_mlps_external = pretrain_alignment_mlp_by_stats_v2(
        inc_feats_raw,  # 源特征 (新任务)
        old_feats,  # 目标特征 (旧任务)
        device,
        epochs=500,
        lr=0.002
    )

    with torch.no_grad():
        inc_feats_aligned_externally = [
            mlp(torch.tensor(f, dtype=torch.float32, device=device)).cpu().numpy()
            for mlp, f in zip(align_mlps_external, inc_feats_raw)
        ]
    inc_drug_fg_aligned, inc_drug_features_aligned, inc_drug_bert_aligned, \
        inc_microbe_features_aligned, inc_microbe_bert_aligned, inc_microbe_path_aligned = inc_feats_aligned_externally
    print(f"外部对齐完成。{inc_dataset_name} 特征维度已对齐到 {args.dataset} 空间。")

    # --- 步骤 4: 【内部对齐】创建一组新的、未训练的MLP用于端到端微调 ---
    print("步骤 3/5: 创建【内部对齐】所需的MLP模块...")
    internal_align_mlps = train_alignment_mlps(old_feats, old_feats, device)  # 输入输出维度都和旧任务一致

    # --- 步骤 5: 加载新任务的图结构和标签数据 ---
    print(f"步骤 4/5: 加载 {inc_dataset_name} 的图结构和标签...")
    inc_Sd = np.loadtxt(inc_base_path + 'drugsimilarity.txt')
    inc_Sm = np.loadtxt(inc_base_path + 'microbesimilarity.txt')
    df_inc = pd.read_csv(inc_base_path + 'adj_out.txt', sep='\s+', header=None, names=['drug', 'microbe', 'value'])
    num_drug_inc = df_inc['drug'].max()
    num_microbe_inc = df_inc['microbe'].max()
    adj_inc = np.zeros((num_drug_inc + 1, num_microbe_inc + 1), dtype=int)
    for _, row in df_inc.iterrows():
        adj_inc[row['drug'], row['microbe']] = row['value']
    I_inc = adj_inc
    pos_inc, neg_inc = get_positive_negative_samples(adj_inc)

    # --- 步骤 6: 准备5折交叉验证的循环 ---
    print("步骤 5/5: 准备进入5折交叉验证微调循环...")
    from sklearn.model_selection import train_test_split
    from numpy.random import default_rng

    rng = default_rng(42)
    auc_list_inc, aupr_list_inc, acc_list_inc = [], [], []
    auc_list_old, aupr_list_old, acc_list_old = [], [], []

    # ==============================================================================
    #  注意: 从这里开始，你需要将原代码中所有的 `drugvirus_` 或 `_dv` 变量前缀
    #  替换为通用的 `inc_` (代表 incremental)。
    #  例如： `pos_dv` -> `pos_inc`, `drugvirus_train_data` -> `inc_train_data`, etc.
    #  下面的代码已经帮你完成了这个替换。
    # ==============================================================================

    for finetune_fold in range(5):
        print(f"\n========== 微调 fold {finetune_fold + 1}/5 ==========")

        # 划分新任务数据
        pos_train_and_val, pos_test = train_test_split(pos_inc, test_size=0.2, random_state=42 + finetune_fold,
                                                       shuffle=True)
        neg_train_and_val, neg_test = train_test_split(neg_inc, test_size=0.2, random_state=42 + finetune_fold,
                                                       shuffle=True)
        neg_test_balanced = neg_test[rng.choice(len(neg_test), size=len(pos_test), replace=False)]
        test_samples_inc = np.vstack([pos_test, neg_test_balanced])
        test_labels_inc = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test_balanced))])
        test_drug_idx_inc = test_samples_inc[:, 0].astype(int)
        test_microbe_idx_inc = test_samples_inc[:, 1].astype(int)
        inc_test_data = (test_drug_idx_inc, test_microbe_idx_inc, test_labels_inc)

        neg_train_and_val_balanced = neg_train_and_val[
            rng.choice(len(neg_train_and_val), size=len(pos_train_and_val), replace=False)]
        train_val_samples_inc = np.vstack([pos_train_and_val, neg_train_and_val_balanced])
        train_val_labels_inc = np.concatenate(
            [np.ones(len(pos_train_and_val)), np.zeros(len(neg_train_and_val_balanced))])
        train_samples_new_inc, val_samples_inc, train_labels_new_inc, val_labels_inc = train_test_split(
            train_val_samples_inc, train_val_labels_inc, test_size=len(test_samples_inc),
            random_state=42 + finetune_fold, stratify=train_val_labels_inc
        )
        train_drug_idx_new_inc = train_samples_new_inc[:, 0].astype(int)
        train_microbe_idx_new_inc = train_samples_new_inc[:, 1].astype(int)
        inc_train_data = (train_drug_idx_new_inc, train_microbe_idx_new_inc, train_labels_new_inc)

        # 特征归一化
        from sklearn.preprocessing import StandardScaler

        scaler_fg_inc = StandardScaler().fit(inc_drug_fg_aligned[train_drug_idx_new_inc])
        scaler_feat_inc = StandardScaler().fit(inc_drug_features_aligned[train_drug_idx_new_inc])
        scaler_bert_inc = StandardScaler().fit(inc_drug_bert_aligned[train_drug_idx_new_inc])
        inc_drug_fg_norm = scaler_fg_inc.transform(inc_drug_fg_aligned)
        inc_drug_features_norm = scaler_feat_inc.transform(inc_drug_features_aligned)
        inc_drug_bert_norm = scaler_bert_inc.transform(inc_drug_bert_aligned)
        scaler_microbe_feat_inc = StandardScaler().fit(inc_microbe_features_aligned[train_microbe_idx_new_inc])
        scaler_microbe_bert_inc = StandardScaler().fit(inc_microbe_bert_aligned[train_microbe_idx_new_inc])
        scaler_microbe_path_inc = StandardScaler().fit(inc_microbe_path_aligned[train_microbe_idx_new_inc])
        inc_microbe_features_norm = scaler_microbe_feat_inc.transform(inc_microbe_features_aligned)
        inc_microbe_bert_norm = scaler_microbe_bert_inc.transform(inc_microbe_bert_aligned)
        inc_microbe_path_norm = scaler_microbe_path_inc.transform(inc_microbe_path_aligned)

        # 构建图结构
        I_train_inc = I_inc.copy()
        for d, m in pos_test: I_train_inc[d, m] = 0
        val_pos_mask_inc = (val_labels_inc == 1)
        val_pos_inc = val_samples_inc[val_pos_mask_inc]
        for d, m in val_pos_inc: I_train_inc[d, m] = 0
        A_fold_inc = build_gcn_adj(inc_Sd, inc_Sm, I_train_inc)
        row_inc, col_inc = np.where(A_fold_inc != 0)
        edge_index_inc = torch.tensor(np.stack([row_inc, col_inc]), dtype=torch.long, device=device)
        edge_weight_inc = torch.tensor(A_fold_inc[row_inc, col_inc], dtype=torch.float32, device=device)

        # 加载旧任务模型
        old_task_model_path = f'./{args.dataset}/fold{finetune_fold + 1}/{args.dataset}_hd{args.hidden_dim_retrain}_gcn_model_fused.pth'
        old_task_decoder_path = f'./{args.dataset}/fold{finetune_fold + 1}/{args.dataset}_hd{args.hidden_dim_retrain}_decoder_fused.pth'

        # 根据旧任务设置模型输入输出维度
        if args.dataset == "MDAD":
            drug_out_dim = 1373
        elif args.dataset == 'aBiofilm':
            drug_out_dim = 1720
        elif args.dataset == 'DrugVirus':
            drug_out_dim = 175

        inc_model = GCNWithMLP(
            drug_in_dim=2048, drug_out_dim=drug_out_dim,
            microbe_dim=old_microbe_features.shape[1], microbe_out_dim=old_microbe_features.shape[1],
            gcn_hidden=args.hidden_dim_retrain, dropout=args.dropout, use_microbe_mlp=False,
            dataset_name=args.dataset
        )
        inc_decoder = MLPDecoder(args.hidden_dim_retrain)

        # 加载旧任务模型权重
        if os.path.exists(old_task_model_path): inc_model.load_state_dict(
            torch.load(old_task_model_path, map_location=device), strict=False)
        if os.path.exists(old_task_decoder_path): inc_decoder.load_state_dict(
            torch.load(old_task_decoder_path, map_location=device))
        inc_model = inc_model.to(device)
        inc_decoder = inc_decoder.to(device)

        # 加载旧任务的 Fisher 信息和旧参数
        fisher_path = f'./{args.dataset}/fold{finetune_fold + 1}/fisher.pth'
        fisher_decoder_path = f'./{args.dataset}/fold{finetune_fold + 1}/fisher_decoder.pth'
        old_params_path = f'./{args.dataset}/fold{finetune_fold + 1}/old_params.pth'
        old_params_decoder_path = f'./{args.dataset}/fold{finetune_fold + 1}/old_params_decoder.pth'
        fisher = torch.load(fisher_path, map_location=device)
        fisher_decoder = torch.load(fisher_decoder_path, map_location=device)
        old_params = torch.load(old_params_path, map_location=device)
        old_params_decoder = torch.load(old_params_decoder_path, map_location=device)

        # 加载并重建当前 fold 的旧任务数据 (用于排练)
        old_task_data_path = f'./{args.dataset}/fold{finetune_fold + 1}/{args.dataset}_fold_data_for_ewc.npz'
        loaded_data = np.load(old_task_data_path)
        old_train_data = (loaded_data['drug_idx'], loaded_data['microbe_idx'], loaded_data['labels'])
        adj_matrix_loaded = loaded_data['adj_matrix']
        row, col = np.where(adj_matrix_loaded != 0)
        old_edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
        old_edge_weight = torch.tensor(adj_matrix_loaded[row, col], dtype=torch.float32, device=device)
        old_drug_fg_norm = loaded_data['drug_fg_norm']
        old_drug_features_norm = loaded_data['drug_features_norm']
        old_drug_bert_norm = loaded_data['drug_bert_norm']
        old_microbe_features_norm = loaded_data['microbe_features_norm']
        old_microbe_bert_norm = loaded_data['microbe_bert_norm']
        old_microbe_path_norm = loaded_data['microbe_path_norm']

        # 加载旧任务的测试集 (用于绘图和回测)
        old_test_set_path = f'./{args.dataset}/fold{finetune_fold + 1}/{args.dataset}_test_set.npz'
        old_test_set = np.load(old_test_set_path)
        old_test_data = (old_test_set['test_drug_idx'], old_test_set['test_microbe_idx'], old_test_set['test_labels'])

        # 微调（EWC）
        lambda_ewc = 1e8;
        ewc_lr = 1e-4;
        ewc_epoch = 200;
        alpha = 0.9

        inc_model, inc_decoder = train_gcn_ewc_new(
            # 新任务数据
            inc_train_data, edge_index_inc, edge_weight_inc,
            inc_drug_fg_norm, inc_drug_features_norm, inc_drug_bert_norm,
            inc_microbe_features_norm, inc_microbe_bert_norm, inc_microbe_path_norm, inc_microbe_offset,
            # 旧任务数据 (用于排练)
            old_test_data, old_train_data, old_edge_index, old_edge_weight,
            old_drug_fg_norm, old_drug_features_norm, old_drug_bert_norm,
            old_microbe_features_norm, old_microbe_bert_norm, old_microbe_path_norm, microbe_offset,
            # 超参数
            epochs=ewc_epoch, lr=ewc_lr, hidden=args.hidden_dim_retrain, dropout=args.dropout, device=device,
            old_params=old_params, old_params_decoder=old_params_decoder, fisher=fisher, fisher_decoder=fisher_decoder,
            lambda_ewc=lambda_ewc,
            model=inc_model, decoder=inc_decoder, args=args,
            alignment_mlps=internal_align_mlps, use_feature_alignment=not args.no_feature_alignment,
            alpha=alpha, weight_decay=0,
            # 绘图用数据
            drugvirus_test_data=inc_test_data, fold_num=finetune_fold, save_dir=f'./{args.dataset}/'
        )

        # --- 评估新任务性能 ---
        # (和原代码逻辑一致，只是变量名换了)
        with torch.no_grad():
            align_mlps_external.eval()
            inc_model.eval()
            inc_decoder.eval()
            inc_raw_feats_t = [torch.tensor(f, dtype=torch.float32, device=device) for f in inc_feats_raw]
            externally_aligned_t = [mlp(feat) for mlp, feat in zip(align_mlps_external, inc_raw_feats_t)]
            externally_aligned_np = [t.cpu().numpy() for t in externally_aligned_t]
            externally_aligned_and_norm_np = [
                scaler_fg_inc.transform(externally_aligned_np[0]), scaler_feat_inc.transform(externally_aligned_np[1]),
                scaler_bert_inc.transform(externally_aligned_np[2]),
                scaler_microbe_feat_inc.transform(externally_aligned_np[3]),
                scaler_microbe_bert_inc.transform(externally_aligned_np[4]),
                scaler_microbe_path_inc.transform(externally_aligned_np[5]),
            ]
            final_aligned_feats_for_eval = [f for f in externally_aligned_and_norm_np]

        auc_inc, aupr_inc, acc_inc = evaluate_gcn(
            inc_model, inc_decoder, inc_test_data, edge_index_inc, edge_weight_inc,
            *final_aligned_feats_for_eval, inc_microbe_offset, device=device
        )
        print(
            f'[{inc_dataset_name} 微调后 fold{finetune_fold + 1}] AUC: {auc_inc:.4f}, AUPR: {aupr_inc:.4f}, ACC: {acc_inc:.4f}')
        auc_list_inc.append(auc_inc);
        aupr_list_inc.append(aupr_inc);
        acc_list_inc.append(acc_inc)

        # --- 回测旧任务性能 ---
        # (和原代码逻辑一致，只是变量名换了)
        old_Sd, old_Sm, old_df = Sd, Sm, df
        I_old = old_df.to_numpy()  # 简化处理，实际上你需要根据df正确构建邻接矩阵
        I_old = adj.copy()  # 假设adj是旧任务的
        for d, m in zip(old_test_data[0], old_test_data[1]): I_old[d, m] = 0
        A_fold_old = build_gcn_adj(old_Sd, old_Sm, I_old)
        row_old, col_old = np.where(A_fold_old != 0)
        edge_index_old = torch.tensor(np.stack([row_old, col_old]), dtype=torch.long, device=device)
        edge_weight_old = torch.tensor(A_fold_old[row_old, col_old], dtype=torch.float32, device=device)

        auc_old, aupr_old, acc_old = evaluate_gcn(
            inc_model, inc_decoder, old_test_data, edge_index_old, edge_weight_old,
            old_drug_fg_norm, old_drug_features_norm, old_drug_bert_norm,
            old_microbe_features_norm, old_microbe_bert_norm, old_microbe_path_norm, microbe_offset, device=device
        )
        print(
            f'[{args.dataset} 回测 fold{finetune_fold + 1}] AUC: {auc_old:.4f}, AUPR: {aupr_old:.4f}, ACC: {acc_old:.4f}')
        auc_list_old.append(auc_old);
        aupr_list_old.append(aupr_old);
        acc_list_old.append(acc_old)

    # 5折平均结果
    if auc_list_inc:
        print(
            f"\n==== 5折 {inc_dataset_name} 微调后平均 AUC: {np.mean(auc_list_inc):.4f} ± {np.std(auc_list_inc):.4f}, AUPR: {np.mean(aupr_list_inc):.4f} ± {np.std(aupr_list_inc):.4f}, ACC: {np.mean(acc_list_inc):.4f} ± {np.std(acc_list_inc):.4f} ====")
    if auc_list_old:
        print(
            f"==== 5折 {args.dataset} 回测后平均 AUC: {np.mean(auc_list_old):.4f} ± {np.std(auc_list_old):.4f}, AUPR: {np.mean(aupr_list_old):.4f} ± {np.std(aupr_list_old):.4f}, ACC: {np.mean(acc_list_old):.4f} ± {np.std(acc_list_old):.4f} ====")
    print("==== 增量学习流程结束 ====")



