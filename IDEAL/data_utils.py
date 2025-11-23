from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.model_selection import KFold
import pandas as pd
# def load_features(DRUG_FEATURE_PATH,DRUG_BERT_PATH,DRUG_FG_PATH,MICROBE_FEATURE_PATH,MICROBE_BERT_PATH,MICROBE_PATH_PATH):
#     drug_features = np.loadtxt(DRUG_FEATURE_PATH)
#     drug_bert= pd.read_excel(DRUG_BERT_PATH, header=None, index_col=None).values
#     drug_fg = pd.read_excel(DRUG_FG_PATH, header=None, index_col=None).values
#     #microbe_features = np.loadtxt(microbe_path)
#     # === 新增：三组药物特征分别做标准化 ===
#     # scaler1 = StandardScaler()
#     # scaler2 = StandardScaler()
#     # scaler3 = StandardScaler()
#     # drug_fg = scaler1.fit_transform(drug_fg)
#     # drug_feature = scaler2.fit_transform(drug_feature)
#     # drug_bert = scaler3.fit_transform(drug_bert)
#     # # ================================
#     microbe_features = np.loadtxt(MICROBE_FEATURE_PATH)
#     microbe_bert = pd.read_excel(MICROBE_BERT_PATH, header=None, index_col=None).values
#     microbe_path = pd.read_excel(MICROBE_PATH_PATH, header=None, index_col=None).values
#     return drug_features, drug_bert,drug_fg,microbe_features,microbe_bert,microbe_path
def load_features(DRUG_FEATURE_PATH, DRUG_BERT_PATH, DRUG_FG_PATH, MICROBE_FEATURE_PATH, MICROBE_BERT_PATH,
                  MICROBE_PATH_PATH):
    def process_file(original_path, loader_type):
        """
        内部辅助函数，用于加载单个文件。
        优先读取.npy缓存，如果不存在则从原始文件加载并创建缓存。
        """
        # 构造.npy缓存文件路径，例如 '.../drug_bert.xlsx' -> '.../drug_bert.npy'
        npy_path = os.path.splitext(original_path)[0] + '.npy'

        # 检查.npy文件是否存在
        if os.path.exists(npy_path):
            print(f"成功: 正在从缓存加载 -> {npy_path}")
            return np.load(npy_path)
        else:
            print(f"提示: 未找到缓存文件 {npy_path}。")
            print(f"      正在从原始文件加载 -> {original_path}")

            # 根据文件类型选择加载方式
            if loader_type == 'loadtxt':
                data = np.loadtxt(original_path)
            elif loader_type == 'excel':
                data = pd.read_excel(original_path, header=None, index_col=None).values
            else:
                raise ValueError(f"不支持的加载类型: {loader_type}")

            # 将加载的数据保存为.npy文件，以便下次快速读取
            np.save(npy_path, data)
            print(f"      已创建缓存文件 -> {npy_path}")
            return data

    # 使用辅助函数处理每一个特征文件
    drug_features = process_file(DRUG_FEATURE_PATH, 'loadtxt')
    drug_bert = process_file(DRUG_BERT_PATH, 'excel')
    drug_fg = process_file(DRUG_FG_PATH, 'excel')

    microbe_features = process_file(MICROBE_FEATURE_PATH, 'loadtxt')
    microbe_bert = process_file(MICROBE_BERT_PATH, 'excel')
    microbe_path = process_file(MICROBE_PATH_PATH, 'excel')

    return drug_features, drug_bert, drug_fg, microbe_features, microbe_bert, microbe_path


def load_adj(adj_path):
    adj = np.loadtxt(adj_path)
    return adj

def get_positive_negative_samples(adj):
    pos = np.argwhere(adj == 1)
    # 负样本为adj中值为0的位置（药物-微生物对）
    neg = np.argwhere(adj == 0)
    return pos, neg

def sample_negatives(neg, num_samples, random_state=None):
    np.random.seed(random_state)
    if len(neg) == 0:
        raise ValueError("负样本数量为0，无法采样。请检查邻接矩阵！")
    if num_samples > len(neg):
        num_samples = len(neg)
    idx = np.random.choice(len(neg), num_samples, replace=False)
    return neg[idx]


from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import StratifiedKFold
import numpy as np

import numpy as np
from sklearn.model_selection import KFold

def get_kfold_indices(pos, neg, n_splits=5, random_state=None):
    """
    pos: 正样本数组 shape=(N_pos, 2)
    neg: 负样本数组 shape=(N_neg, 2)
    """
    pos = np.array(pos)
    neg = np.array(neg)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pos_indices = np.arange(len(pos))
    for train_pos_idx, test_pos_idx in kf.split(pos):
        # 正样本
        train_pos = pos[train_pos_idx]
        test_pos = pos[test_pos_idx]
        # 负样本：每折随机采样等数量
        rng = np.random.RandomState(random_state)  # 保持可复现
        train_neg = rng.choice(len(neg), size=len(train_pos), replace=False)
        test_neg = rng.choice(list(set(range(len(neg))) - set(train_neg)), size=len(test_pos), replace=False)
        train_neg = neg[train_neg]
        test_neg = neg[test_neg]
        yield (train_pos, test_pos, train_neg, test_neg)




def build_gcn_adj(Sd, Sm, I):
    """
    构建GCN用的邻接矩阵A，块结构如下：
    A = [[Sd, I],
         [I.T, Sm]]
    """
    n_drug = Sd.shape[0]
    n_microbe = Sm.shape[0]


    top = np.hstack((Sd, I))
    bottom = np.hstack((I.T, Sm))
    A = np.vstack((top, bottom))
    return A


import numpy as np

def normalize(M):
    M = M.astype(float)
    return (M - M.min()) / (M.max() - M.min() + 1e-6)

def fuse_adj_with_mask(Sd, Sm, I, mask, lambda_d=0.8, lambda_m=0.8, lambda_I=0.8):
    n_drug = Sd.shape[0]
    n_microbe = Sm.shape[0]
    # 1. 拆分mask
    M_Sd = mask[:n_drug, :n_drug]
    M_I  = mask[:n_drug, n_drug:]
    M_Sm = mask[n_drug:, n_drug:]

    f=0
    if(f==1):
    # 2. 归一化
        Sd_n = normalize(Sd)
        Sm_n = normalize(Sm)
        I_n  = normalize(I)
        M_Sd_n = normalize(M_Sd)
        M_Sm_n = normalize(M_Sm)
        M_I_n  = normalize(M_I)
    else:
        Sd_n = Sd
        Sm_n = Sm
        I_n = I
        M_Sd_n = M_Sd
        M_Sm_n = M_Sm
        M_I_n = M_I


    # 3. 融合
    Sd_fused = lambda_d * Sd_n + (1 - lambda_d) * M_Sd_n
    Sm_fused = lambda_m * Sm_n + (1 - lambda_m) * M_Sm_n
    I_fused  = lambda_I * I_n  + (1 - lambda_I) * M_I_n

    # Sd_fused = Sd_n + (1 - lambda_d) * M_Sd_n
    # Sm_fused = Sm_n + (1 - lambda_m) * M_Sm_n
    # I_fused =  I_n + (1 - lambda_I) * M_I_n

    # 4. 拼回大邻接矩阵
    top = np.hstack((Sd_fused, I_fused))
    bottom = np.hstack((I_fused.T, Sm_fused))
    A_fused = np.vstack((top, bottom))
    return A_fused
