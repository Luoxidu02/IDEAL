# visualization.py

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import seaborn as sns  # <-- 新增
import pandas as pd    # <-- 新增

# ... (你已有的 plt.rcParams 设置) ...

# 推荐在函数外设置，避免重复加载
# ====================== 推荐的跨平台字体设置 ======================
import platform
import matplotlib.font_manager as fm


def set_chinese_font():
    """
    自动根据操作系统设置 Matplotlib 的中文字体。
    """
    system = platform.system()

    # 字体优先级列表
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'KaiTi']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'Songti SC']
    else:  # Linux 及其他
        font_list = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']

    # 查找系统中可用的字体
    installed_fonts = {f.name for f in fm.fontManager.ttflist}

    # for font_name in font_list:
    #     if font_name in installed_fonts:
    #         plt.rcParams['font.sans-serif'] = [font_name]
    #         print(f"✔ 成功设置字体为: {font_name}")
    #         break
    # else:
    #     # 如果上面的字体都找不到，发出警告
    #     print("⚠ 警告：未找到推荐的中文字体，图形中的中文可能显示为方块。")
    #     print("   - Windows: 请确保已安装 '微软雅黑' 或 '黑体'。")
    #     print("   - macOS: 请确保系统字体库完整。")
    #     print("   - Linux: 请尝试安装 'sudo apt-get install fonts-wqy-zenhei'。")

    # 统一设置负号显示问题
    plt.rcParams['axes.unicode_minus'] = False


# 在所有绘图函数之前调用一次即可
set_chinese_font()


# ================================================================

# 之后的可视化代码保持不变
def visualize_gnnexplainer_subgraph(
        edge_index_used,
        edge_mask,
        target_drug_id,
        target_microbe_id_local,
        microbe_offset,
        fold_dir,
        fold_num,
        top_k=10
):
    """
    Visualize the subgraph for GNNExplainer.

    Args:
        edge_index_used (torch.Tensor or np.ndarray): Subgraph edge indices used by the explainer, shape [2, num_edges].
        edge_mask (torch.Tensor or np.ndarray): Importance scores for each edge, shape [num_edges].
        target_drug_id (int): Target drug's global node ID.
        target_microbe_id_local (int): Target microbe's local ID (column index in adjacency matrix).
        microbe_offset (int): Global offset for microbe node IDs.
        fold_dir (str): Directory to save the image.
        fold_num (int): Current fold number.
        top_k (int): Highlight the top k most important edges.
    """
    # 1. Data preprocessing
    if hasattr(edge_index_used, 'cpu'):
        edge_index_used = edge_index_used.cpu().numpy()
    if hasattr(edge_mask, 'cpu'):
        edge_mask = edge_mask.detach().cpu().numpy()

    target_microbe_id_global = target_microbe_id_local + microbe_offset

    # 2. Create graph object
    G = nx.Graph()
    all_nodes = np.unique(edge_index_used.flatten())
    G.add_nodes_from(all_nodes)

    # 3. Add edges and weights
    edges_with_weights = []
    for i in range(edge_index_used.shape[1]):
        u, v = edge_index_used[0, i], edge_index_used[1, i]
        weight = edge_mask[i]
        # Exclude self-loops
        if u != v:
            G.add_edge(u, v, weight=weight)
            edges_with_weights.append(weight)

    # 4. Set visual attributes for nodes and edges
    node_colors = []
    node_sizes = []
    node_labels = {}
    for node in G.nodes():
        if node == target_drug_id:
            node_colors.append('#ff5733')  # Bright red
            node_sizes.append(500)
            node_labels[node] = f"Drug\n{target_drug_id}"
        elif node == target_microbe_id_global:
            node_colors.append('#33a7ff')  # Bright blue
            node_sizes.append(500)
            node_labels[node] = f"Microbe\n{target_microbe_id_local}"
        else:
            node_colors.append('#cccccc')  # Gray
            node_sizes.append(150)

    # Edge colors and widths
    edge_weights_normalized = [w for u, v, w in G.edges(data='weight')]

    cmap = cm.get_cmap('viridis')
    edge_colors = [cmap(w) for w in edge_weights_normalized]

    edge_widths = [1 + 5 * w for w in edge_weights_normalized]

    # Highlight Top-K edges
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
    top_k_edges = [e[:2] for e in sorted_edges[:top_k]]

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.8)  # Adjust node spacing

    # Draw all nodes and edges as background
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.9, ax=ax, edge_cmap=cmap)

    # Draw highlighted Top-K edges
    nx.draw_networkx_edges(G, pos, edgelist=top_k_edges, width=3.0, edge_color='orange', style='dashed', ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black', font_weight='bold')

    ax.set_title(
        f'GNNExplainer Subgraph for Drug {target_drug_id} & Microbe {target_microbe_id_local} (Fold {fold_num + 1})',
        fontsize=16)
    plt.axis('off')

    # Add a color bar to explain edge importance
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=min(edge_weights_normalized), vmax=max(edge_weights_normalized)))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Edge Importance', rotation=270, labelpad=20)

    # 6. Save the image
    save_path = os.path.join(fold_dir,
                             f'gnnexplainer_fold{fold_num + 1}_drug{target_drug_id}_microbe{target_microbe_id_local}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"★★ GNNExplainer visualization saved to: {save_path} ★★")


# visualization.py

# ... (文件顶部的其他import和函数保持不变) ...
# visualization.py

# ... (文件顶部的其他import和函数保持不变) ...

# ====================================================================================
# ==============  【【【 终极修复版：使用 a more robust 'mask' 参数】】】 ==============
# ====================================================================================
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


def visualize_as_heatmap(
        edge_index_used,
        edge_mask,
        target_drug_id,
        target_microbe_id_local,
        microbe_offset,
        fold_dir,
        fold_num
):
    """
    将 GNNExplainer 的解释结果可视化为一张美观的热力图。
    【终极修复版】特性：
    - 使用 seaborn 内建的 'mask' 参数，彻底解决全红覆盖问题。
    - "不存在的边" 和 "重要性为0的边" 显示为灰色背景。
    - Top-10 最重要的边用亮红色高亮显示。
    - 其他重要性 > 0 的边使用 viridis 色图。
    """
    # --- 1. 数据预处理 ---
    if hasattr(edge_index_used, 'cpu'):
        edge_index_used = edge_index_used.cpu().numpy()
    if hasattr(edge_mask, 'cpu'):
        edge_mask = edge_mask.detach().cpu().numpy()

    subgraph_nodes = np.unique(edge_index_used.flatten())
    num_nodes_in_subgraph = len(subgraph_nodes)
    node_to_matrix_map = {node_id: i for i, node_id in enumerate(subgraph_nodes)}

    # --- 2. 构建两个未排序的矩阵 ---
    importance_matrix = np.full((num_nodes_in_subgraph, num_nodes_in_subgraph), -1.0)
    for i in range(edge_index_used.shape[1]):
        u, v = edge_index_used[0, i], edge_index_used[1, i]
        importance = edge_mask[i]
        if u in node_to_matrix_map and v in node_to_matrix_map:
            u_idx, v_idx = node_to_matrix_map[u], node_to_matrix_map[v]
            importance_matrix[u_idx, v_idx] = importance
            importance_matrix[v_idx, u_idx] = importance

    highlight_matrix = np.zeros_like(importance_matrix, dtype=float)
    if edge_mask.size > 0:
        num_to_highlight = min(10, edge_mask.size)
        top_k_indices = np.argsort(edge_mask)[-num_to_highlight:]

        for idx in top_k_indices:
            u, v = edge_index_used[0, idx], edge_index_used[1, idx]
            if u in node_to_matrix_map and v in node_to_matrix_map:
                u_idx, v_idx = node_to_matrix_map[u], node_to_matrix_map[v]
                highlight_matrix[u_idx, v_idx] = edge_mask[idx]
                highlight_matrix[v_idx, u_idx] = edge_mask[idx]

    # --- 3. 同时对两个矩阵进行排序 ---
    drug_indices = [i for i, node_id in enumerate(subgraph_nodes) if node_id < microbe_offset]
    microbe_indices = [i for i, node_id in enumerate(subgraph_nodes) if node_id >= microbe_offset]
    reorder_indices = drug_indices + microbe_indices

    importance_matrix = importance_matrix[np.ix_(reorder_indices, reorder_indices)]
    highlight_matrix = highlight_matrix[np.ix_(reorder_indices, reorder_indices)]

    # --- 4. 准备绘图颜色和遮罩 ---

    # a) 基础热力图颜色和Numpy掩码数组
    my_cmap = plt.get_cmap('viridis').copy()
    my_cmap.set_bad(color='#e0e0e0')
    masked_matrix = np.ma.masked_where(importance_matrix <= 0, importance_matrix)

    # b) 高亮热力图颜色
    red_cmap = ListedColormap(['#FF0000'])

    # --- 5. 绘制热力图 (两层叠加) ---
    plt.figure(figsize=(12, 10), dpi=300)

    # 第一层：绘制基础热力图 (灰色背景 + viridis颜色)
    ax = sns.heatmap(
        masked_matrix,
        cmap=my_cmap,
        xticklabels=False,
        yticklabels=False,
        square=True,
        cbar_kws={'shrink': 0.7},
        vmin=np.finfo(float).eps
    )

    # ========================== 【【【 核心修复 】】】 ==========================
    # 第二层：使用'mask'参数叠加高亮热力图
    sns.heatmap(
        highlight_matrix,  # <-- 传入原始的、未被numpy mask的矩阵
        cmap=red_cmap,
        ax=ax,
        cbar=False,
        # 关键！创建一个布尔掩码，所有值为0的格子都将被隐藏
        mask=(highlight_matrix == 0),
        xticklabels=False,
        yticklabels=False,
        square=True
    )
    # =======================================================================

    ax.set_title(
        f'GNNExplainer Importance Heatmap for Drug {target_drug_id} & Microbe {target_microbe_id_local} (Fold {fold_num})',
        fontsize=16, pad=20)

    # 获取并设置颜色条
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.set_label('Edge Importance', fontsize=12)

    plt.tight_layout()

    # --- 6. 保存图像 ---
    heatmap_save_dir = os.path.join(fold_dir, 'heatmaps')
    os.makedirs(heatmap_save_dir, exist_ok=True)

    save_path = os.path.join(heatmap_save_dir,
                             f'gnnexplainer_heatmap_fold{fold_num}_drug{target_drug_id}_microbe{target_microbe_id_local}.png')
    plt.savefig(save_path)
    print(f"★★ GNNExplainer Heatmap saved to: {save_path} ★★")
    plt.close()
