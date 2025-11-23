# # aggregate_visualization.py
#
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import gc
# from tqdm import tqdm
# import platform
# import matplotlib.font_manager as fm
# import pandas as pd  # 新增导入
#
# # 从项目其他文件中导入必要的模块
# from gcn_model import GCNWithDecoderWrapper, GCNWithDecoderWrapperBatched, explain_with_full_graph
# from torch_geometric.explain import Explainer, GNNExplainer
# from torch_geometric.explain.config import ModelConfig, ModelReturnType
#
#
# def set_chinese_font():
#     """
#     自动根据操作系统设置 Matplotlib 的中文字体。
#     """
#     system = platform.system()
#     font_list = ['Microsoft YaHei', 'SimHei'] if system == 'Windows' else ['PingFang SC', 'Heiti SC']
#
#     installed_fonts = {f.name for f in fm.fontManager.ttflist}
#     for font_name in font_list:
#         if font_name in installed_fonts:
#             plt.rcParams['font.sans-serif'] = [font_name]
#             break
#     plt.rcParams['axes.unicode_minus'] = False
#
#
# set_chinese_font()
#
#
# def load_entity_names(dataset_name):
#     """
#     根据数据集名称加载药物和微生物的名称列表。
#     """
#     base_path = f'{dataset_name}'
#     # 在这里定义不同数据集的名称文件路径
#     if dataset_name == 'MDAD':
#         drug_file = os.path.join(base_path, 'drugs.xlsx')
#         microbe_file = os.path.join(base_path, 'microbes.xlsx')
#         drug_name_col = 'Drug_name'
#         microbe_name_col = 'Microbe_name'
#     elif dataset_name == 'aBiofilm':
#         # 假设 aBiofilm 数据集的文件名和列名如下，请根据实际情况修改
#         drug_file = os.path.join(base_path, 'drug_names.xlsx')
#         microbe_file = os.path.join(base_path, 'microbe_names.xlsx')
#         drug_name_col = 'DrugName'
#         microbe_name_col = 'MicrobeName'
#     elif dataset_name == 'DrugVirus':
#         # 假设 aBiofilm 数据集的文件名和列名如下，请根据实际情况修改
#         drug_file = os.path.join(base_path, 'drugs.xlsx')
#         microbe_file = os.path.join(base_path, 'viruses.xlsx')
#         drug_name_col = 'Drugs'
#         microbe_name_col = 'Virus'
#     else:
#         print(f"警告: 未为数据集 '{dataset_name}' 配置名称文件路径。将无法显示具体名称。")
#         return None, None
#
#     try:
#         drug_df = pd.read_excel(drug_file)
#         microbe_df = pd.read_excel(microbe_file)
#
#         # 将名称列转换为列表（0-based index）
#         drug_names = drug_df[drug_name_col].tolist()
#         microbe_names = microbe_df[microbe_name_col].tolist()
#
#         print(f"✔ 成功加载 {len(drug_names)} 个药物名称和 {len(microbe_names)} 个微生物名称。")
#         return drug_names, microbe_names
#     except FileNotFoundError as e:
#         print(f"错误: 找不到名称文件: {e}。请检查文件路径是否正确。")
#         return None, None
#     except KeyError as e:
#         print(f"错误: 在Excel文件中找不到指定的列名: {e}。请检查列名是否正确。")
#         return None, None
#
#
# def run_aggregate_visualization(
#         model,
#         decoder,
#         train_pos_samples_to_explain,
#         embeddings_cached,
#         X,
#         edge_index,
#         edge_weight,
#         I_train,
#         microbe_offset,
#         fold_dir,
#         fold_num,
#         args,
#         device
# ):
#     """
#     执行GNNExplainer的聚合分析，生成条形图，并导出详细的Excel解释报告。
#     """
#     num_samples_to_aggregate = len(train_pos_samples_to_explain)
#     print(f"\n===== [方案三] 开始为 Fold {fold_num} 生成聚合重要性分析 =====")
#     print(f"将对 {num_samples_to_aggregate} 个高置信度正样本进行解释...")
#
#     # --- 新增：加载药物和微生物名称 ---
#     drug_names, microbe_names = load_entity_names(args.dataset)
#     if not drug_names or not microbe_names:
#         print("由于无法加载名称，将跳过生成详细的Excel报告。")
#         names_available = False
#     else:
#         names_available = True
#
#     # 初始化存储结构
#     total_importance = {"药物相似性 (D-D)": 0.0, "微生物相似性 (M-M)": 0.0, "已知关联 (D-M)": 0.0}
#     detailed_explanations = []  # <<< 新增：用于存储详细解释的列表
#
#     for drug_idx, microbe_idx in tqdm(train_pos_samples_to_explain, desc=f"聚合解释 Fold {fold_num}"):
#
#         src = drug_idx
#         dst = microbe_idx + microbe_offset
#
#         # 同样使用原有的解释器设置和执行逻辑
#         if args.dataset in ['aBiofilm', 'MDAD']:
#             wrapped_model = GCNWithDecoderWrapperBatched(gcn_model=model, decoder=decoder,
#                                                          microbe_offset=microbe_offset, batch_size=4096).to(device)
#         else:
#             wrapped_model = GCNWithDecoderWrapper(model, decoder, microbe_offset).to(device)
#         wrapped_model.eval()
#
#         explainer = Explainer(
#             model=wrapped_model, algorithm=GNNExplainer(epochs=100),
#             explanation_type='phenomenon', edge_mask_type='object',
#             model_config=ModelConfig(mode='binary_classification', task_level='edge', return_type=ModelReturnType.raw),
#         )
#
#         row_np, col_np = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
#         labels = [int(I_train[s, d - microbe_offset] == 1) if s < microbe_offset and d >= microbe_offset else 0 for s, d
#                   in zip(row_np, col_np)]
#         edge_labels = torch.tensor(labels, dtype=torch.long, device=X.device)
#
#         try:
#             explanation, edge_index_used = explain_with_full_graph(explainer, X, edge_index, edge_weight, src, dst,
#                                                                    edge_labels)
#
#             edge_mask = explanation.edge_mask.detach().cpu().numpy()
#             edge_index_np = edge_index_used.cpu().numpy()
#
#             importance_threshold = 0.01
#
#             # --- 改造：分类、累加并记录详细信息 ---
#             for i in range(len(edge_mask)):
#                 if edge_mask[i] > importance_threshold:
#                     u, v = edge_index_np[0, i], edge_index_np[1, i]
#                     importance_score = edge_mask[i]
#
#                     is_u_drug = u < microbe_offset
#                     is_v_drug = v < microbe_offset
#
#                     record = None
#
#                     if is_u_drug and is_v_drug:
#                         edge_type = "药物相似性 (D-D)"
#                         total_importance[edge_type] += importance_score
#                         if names_available:
#                             record = {
#                                 "被解释的预测": f"{drug_names[drug_idx]} - {microbe_names[microbe_idx]}",
#                                 "重要边类型": edge_type,
#                                 "节点1": drug_names[u],
#                                 "节点2": drug_names[v],
#                                 "重要性得分": importance_score
#                             }
#                     elif not is_u_drug and not is_v_drug:
#                         edge_type = "微生物相似性 (M-M)"
#                         total_importance[edge_type] += importance_score
#                         if names_available:
#                             record = {
#                                 "被解释的预测": f"{drug_names[drug_idx]} - {microbe_names[microbe_idx]}",
#                                 "重要边类型": edge_type,
#                                 "节点1": microbe_names[u - microbe_offset],
#                                 "节点2": microbe_names[v - microbe_offset],
#                                 "重要性得分": importance_score
#                             }
#                     else:
#                         edge_type = "已知关联 (D-M)"
#                         total_importance[edge_type] += importance_score
#                         if names_available:
#                             # 确保节点1是药物，节点2是微生物
#                             node1_idx = u if is_u_drug else v
#                             node2_idx = v if is_u_drug else u
#                             record = {
#                                 "被解释的预测": f"{drug_names[drug_idx]} - {microbe_names[microbe_idx]}",
#                                 "重要边类型": edge_type,
#                                 "节点1": drug_names[node1_idx],
#                                 "节点2": microbe_names[node2_idx - microbe_offset],
#                                 "重要性得分": importance_score
#                             }
#
#                     if record:
#                         detailed_explanations.append(record)
#
#         except Exception as e:
#             print(f"\n警告: 在解释边 {src}->{dst} 时发生错误，跳过此样本。错误: {e}")
#             continue
#         finally:
#             del explainer, wrapped_model, explanation
#             gc.collect()
#             torch.cuda.empty_cache()
#
#     # --- 生成聚合条形图 (逻辑不变) ---
#     if num_samples_to_aggregate > 0:
#         avg_importance = {key: val / num_samples_to_aggregate for key, val in total_importance.items()}
#         labels, values = list(avg_importance.keys()), list(avg_importance.values())
#
#         fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
#         colors = ['#ff9999', '#66b3ff', '#99ff99']
#         bars = ax.bar(labels, values, color=colors, edgecolor='black')
#
#         ax.set_ylabel('平均重要性得分 (Average Importance)', fontsize=12)
#         ax.set_title(f'模型决策依据的聚合分析 (Fold {fold_num})\n(基于 {num_samples_to_aggregate} 个样本的解释)',
#                      fontsize=14)
#         ax.set_ylim(0, max(values) * 1.15 if values and max(values) > 0 else 1)
#         for bar in bars:
#             yval = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width() / 2.0, yval + max(values) * 0.02, f'{yval:.3f}', ha='center',
#                     va='bottom', fontsize=10)
#
#         plt.tight_layout()
#         save_path = os.path.join(fold_dir, f'aggregate_importance_fold_{fold_num}.png')
#         plt.savefig(save_path)
#         print(f"\n★★ [方案三] 聚合重要性条形图已保存到: {save_path} ★★")
#         plt.close(fig)
#     else:
#         print("警告: 没有成功解释任何样本，无法生成聚合图。")
#
#     # --- 新增：保存详细解释到Excel文件 ---
#     if names_available and detailed_explanations:
#         df = pd.DataFrame(detailed_explanations)
#
#         # 按重要性得分降序排序
#         df_sorted = df.sort_values(by="重要性得分", ascending=False)
#
#         # 定义Excel保存路径
#         excel_save_path = os.path.join(fold_dir, f'aggregate_explanation_details_fold_{fold_num}.xlsx')
#
#         try:
#             # 保存到Excel
#             df_sorted.to_excel(excel_save_path, index=False, engine='openpyxl')
#             print(f"★★ [方案三] 详细解释报告已保存到: {excel_save_path} ★★")
#             print("您现在可以打开此Excel文件，查看对模型预测贡献最大的具体边。")
#         except Exception as e:
#             print(f"错误: 保存Excel文件失败: {e}")
#     elif not detailed_explanations:
#         print("提示: 未找到重要性得分高于阈值的边，因此不生成详细的Excel报告。")
#
# aggregate_visualization.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from tqdm import tqdm
import platform
import matplotlib.font_manager as fm
import pandas as pd

# 从项目其他文件中导入必要的模块
from gcn_model import GCNWithDecoderWrapper, GCNWithDecoderWrapperBatched, explain_with_full_graph
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelReturnType


def set_chinese_font():
    """
    自动根据操作系统设置 Matplotlib 的中文字体。
    """
    system = platform.system()
    font_list = ['Microsoft YaHei', 'SimHei'] if system == 'Windows' else ['PingFang SC', 'Heiti SC']

    installed_fonts = {f.name for f in fm.fontManager.ttflist}
    for font_name in font_list:
        if font_name in installed_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    plt.rcParams['axes.unicode_minus'] = False


set_chinese_font()


def load_entity_names(dataset_name):
    """
    根据数据集名称加载药物和微生物的名称列表。
    """
    base_path = f'{dataset_name}'
    # 在这里定义不同数据集的名称文件路径
    if dataset_name == 'MDAD':
        drug_file = os.path.join(base_path, 'drugs.xlsx')
        microbe_file = os.path.join(base_path, 'microbes.xlsx')
        drug_name_col = 'Drug_name'
        microbe_name_col = 'Microbe_name'
    elif dataset_name == 'aBiofilm':
        # 假设 aBiofilm 数据集的文件名和列名如下，请根据实际情况修改
        drug_file = os.path.join(base_path, 'drug_names.xlsx')
        microbe_file = os.path.join(base_path, 'microbe_names.xlsx')
        drug_name_col = 'DrugName'
        microbe_name_col = 'MicrobeName'
    elif dataset_name == 'DrugVirus':
        # 假设 aBiofilm 数据集的文件名和列名如下，请根据实际情况修改
        drug_file = os.path.join(base_path, 'drugs.xlsx')
        microbe_file = os.path.join(base_path, 'viruses.xlsx')
        drug_name_col = 'Drugs'
        microbe_name_col = 'Virus'
    else:
        print(f"警告: 未为数据集 '{dataset_name}' 配置名称文件路径。将无法显示具体名称。")
        return None, None

    try:
        drug_df = pd.read_excel(drug_file)
        microbe_df = pd.read_excel(microbe_file)

        # 将名称列转换为列表（0-based index）
        drug_names = drug_df[drug_name_col].tolist()
        microbe_names = microbe_df[microbe_name_col].tolist()

        print(f"✔ 成功加载 {len(drug_names)} 个药物名称和 {len(microbe_names)} 个微生物名称。")
        return drug_names, microbe_names
    except FileNotFoundError as e:
        print(f"错误: 找不到名称文件: {e}。请检查文件路径是否正确。")
        return None, None
    except KeyError as e:
        print(f"错误: 在Excel文件中找不到指定的列名: {e}。请检查列名是否正确。")
        return None, None


def run_aggregate_visualization(
        model,
        decoder,
        train_pos_samples_to_explain,
        embeddings_cached,
        X,
        edge_index,
        edge_weight,
        I_train,
        microbe_offset,
        fold_dir,
        fold_num,
        args,
        device
):
    """
    执行GNNExplainer的聚合分析，生成条形图，并导出详细的Excel解释报告。
    """
    num_samples_to_aggregate = len(train_pos_samples_to_explain)
    print(f"\n===== [方案三] 开始为 Fold {fold_num} 生成聚合重要性分析 =====")
    print(f"将对 {num_samples_to_aggregate} 个高置信度正样本进行解释...")

    # --- 加载药物和微生物名称 ---
    drug_names, microbe_names = load_entity_names(args.dataset)
    if not drug_names or not microbe_names:
        print("由于无法加载名称，将跳过生成详细的Excel报告。")
        names_available = False
    else:
        names_available = True

    # 初始化存储结构
    total_importance = {"药物相似性 (D-D)": 0.0, "微生物相似性 (M-M)": 0.0, "已知关联 (D-M)": 0.0}
    detailed_explanations = []

    for drug_idx, microbe_idx in tqdm(train_pos_samples_to_explain, desc=f"聚合解释 Fold {fold_num}"):

        src = drug_idx
        dst = microbe_idx + microbe_offset

        if args.dataset in ['aBiofilm', 'MDAD']:
            wrapped_model = GCNWithDecoderWrapperBatched(gcn_model=model, decoder=decoder,
                                                         microbe_offset=microbe_offset, batch_size=4096).to(device)
        else:
            wrapped_model = GCNWithDecoderWrapper(model, decoder, microbe_offset).to(device)
        wrapped_model.eval()

        explainer = Explainer(
            model=wrapped_model, algorithm=GNNExplainer(epochs=100),
            explanation_type='phenomenon', edge_mask_type='object',
            model_config=ModelConfig(mode='binary_classification', task_level='edge', return_type=ModelReturnType.raw),
        )

        row_np, col_np = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        labels = [int(I_train[s, d - microbe_offset] == 1) if s < microbe_offset and d >= microbe_offset else 0 for s, d
                  in zip(row_np, col_np)]
        edge_labels = torch.tensor(labels, dtype=torch.long, device=X.device)

        try:
            explanation, edge_index_used = explain_with_full_graph(explainer, X, edge_index, edge_weight, src, dst,
                                                                   edge_labels)

            edge_mask = explanation.edge_mask.detach().cpu().numpy()
            edge_index_np = edge_index_used.cpu().numpy()

            importance_threshold = 0.01

            for i in range(len(edge_mask)):
                if edge_mask[i] > importance_threshold:
                    u, v = edge_index_np[0, i], edge_index_np[1, i]
                    importance_score = edge_mask[i]

                    is_u_drug = u < microbe_offset
                    is_v_drug = v < microbe_offset

                    record = None

                    if is_u_drug and is_v_drug:
                        edge_type = "药物相似性 (D-D)"
                        total_importance[edge_type] += importance_score
                        if names_available:
                            record = {
                                "被解释的预测": f"{drug_names[drug_idx]} - {microbe_names[microbe_idx]}",
                                "重要边类型": edge_type,
                                "节点1": drug_names[u],
                                "节点2": drug_names[v],
                                "重要性得分": importance_score
                            }
                    elif not is_u_drug and not is_v_drug:
                        edge_type = "微生物相似性 (M-M)"
                        total_importance[edge_type] += importance_score
                        if names_available:
                            record = {
                                "被解释的预测": f"{drug_names[drug_idx]} - {microbe_names[microbe_idx]}",
                                "重要边类型": edge_type,
                                "节点1": microbe_names[u - microbe_offset],
                                "节点2": microbe_names[v - microbe_offset],
                                "重要性得分": importance_score
                            }
                    else:
                        edge_type = "已知关联 (D-M)"
                        total_importance[edge_type] += importance_score
                        if names_available:
                            node1_idx = u if is_u_drug else v
                            node2_idx = v if is_u_drug else u
                            record = {
                                # 被解释的预测": f"{drug_names[drug_idx]} - {microbe_names[microbe_idx]}",
                                "重要边类型": edge_type,
                                "节点1": drug_names[node1_idx],
                                "节点2": microbe_names[node2_idx - microbe_offset],
                                "重要性得分": importance_score
                            }

                    if record:
                        detailed_explanations.append(record)

        except Exception as e:
            print(f"\n警告: 在解释边 {src}->{dst} 时发生错误，跳过此样本。错误: {e}")
            continue
        finally:
            del explainer, wrapped_model, explanation
            gc.collect()
            torch.cuda.empty_cache()

    # --- 生成聚合条形图 (逻辑不变) ---
    if num_samples_to_aggregate > 0:
        avg_importance = {key: val / num_samples_to_aggregate for key, val in total_importance.items()}
        labels, values = list(avg_importance.keys()), list(avg_importance.values())

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        bars = ax.bar(labels, values, color=colors, edgecolor='black')

        ax.set_ylabel('平均重要性得分 (Average Importance)', fontsize=12)
        ax.set_title(f'模型决策依据的聚合分析 (Fold {fold_num})\n(基于 {num_samples_to_aggregate} 个样本的解释)',
                     fontsize=14)
        ax.set_ylim(0, max(values) * 1.15 if values and max(values) > 0 else 1)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + max(values) * 0.02, f'{yval:.3f}', ha='center',
                    va='bottom', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(fold_dir, f'aggregate_importance_fold_{fold_num}.png')
        plt.savefig(save_path)
        print(f"\n★★ [方案三] 聚合重要性条形图已保存到: {save_path} ★★")
        plt.close(fig)
    else:
        print("警告: 没有成功解释任何样本，无法生成聚合图。")

    # --- 修改部分：根据数据集类型，有选择地保存详细解释到Excel文件 ---
    if names_available and detailed_explanations:
        df = pd.DataFrame(detailed_explanations)

        # 统一按重要性得分降序排序
        df_sorted = df.sort_values(by="重要性得分", ascending=False)

        df_to_save = None  # 初始化一个用于保存的DataFrame

        # >>>>>>>>>>>>>>>>>>>> 开始修改 <<<<<<<<<<<<<<<<<<<<
        if args.dataset == 'MDAD':
            print("\n为 MDAD 数据集进行特殊处理：筛选每个类别中最重要的边以保存到Excel...")

            # 1. 按类型筛选
            df_dd = df_sorted[df_sorted['重要边类型'] == '药物相似性 (D-D)']
            df_dm = df_sorted[df_sorted['重要边类型'] == '已知关联 (D-M)']
            df_mm = df_sorted[df_sorted['重要边类型'] == '微生物相似性 (M-M)']

            # 2. 提取每个类别的前N条
            top_dd = df_dd.head(300)
            top_dm = df_dm.head(200)
            top_mm = df_mm.head(300)

            print(f"  - 已选择 D-D 边: {len(top_dd)} / {len(df_dd)} 条 (最多300条)")
            print(f"  - 已选择 D-M 边: {len(top_dm)} / {len(df_dm)} 条 (最多200条)")
            print(f"  - 已选择 M-M 边: {len(top_mm)} / {len(df_mm)} 条 (最多300条)")

            # 3. 合并成最终的DataFrame
            df_to_save = pd.concat([top_dd, top_dm, top_mm])

        else:
            # 对于其他数据集，保存所有边 (原始逻辑)
            df_to_save = df_sorted
        # >>>>>>>>>>>>>>>>>>>> 结束修改 <<<<<<<<<<<<<<<<<<<<

        if df_to_save is not None and not df_to_save.empty:
            excel_save_path = os.path.join(fold_dir, f'aggregate_explanation_details_fold_{fold_num}.xlsx')
            try:
                # 检查行数是否会超过Excel限制
                if len(df_to_save) > 1048575:
                    print(f"警告: 筛选后的数据仍有 {len(df_to_save)} 行，超过Excel单工作表最大行数限制。")
                    print("将自动切换为CSV格式进行保存。")
                    csv_save_path = excel_save_path.replace('.xlsx', '.csv')
                    df_to_save.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
                    print(f"★★ [方案三] 详细解释报告已保存为CSV文件: {csv_save_path} ★★")
                else:
                    # 保存到Excel
                    df_to_save.to_excel(excel_save_path, index=False, engine='openpyxl')
                    print(f"★★ [方案三] 详细解释报告已保存到: {excel_save_path} ★★")
                    print("您现在可以打开此Excel文件，查看对模型预测贡献最大的具体边。")

            except Exception as e:
                print(f"错误: 保存文件失败: {e}")

    elif not detailed_explanations:
        print("提示: 未找到重要性得分高于阈值的边，因此不生成详细的Excel报告。")

