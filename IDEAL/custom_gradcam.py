import torch
from torch_geometric.explain import ExplainerAlgorithm
from torch_geometric.nn import GCNConv

# class CustomGradCAM(ExplainerAlgorithm):
#     def __init__(self, target_layer):
#         super().__init__()
#         self.target_layer = target_layer
#         self.activations = None
#         self.gradients = None
#
#     def _save_activations(self, module, input, output):
#         self.activations = output
#
#     def _save_gradients(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]
#
#     def forward(self, model, x, edge_index, *, target, index, edge_weight=None, **kwargs):
#         # 确保模型处于评估模式
#         model.eval()
#
#         # 注册钩子以捕获目标层的激活和梯度
#         # 这里的 target_layer 是一个模型模块，比如 model.conv2
#         forward_handle = self.target_layer.register_forward_hook(self._save_activations)
#         backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)
#
#         try:
#             # 1. 前向传播以获取模型输出
#             # Explainer框架会自动处理这里的target和index，我们只需要正常调用模型
#             # 这里的 x, edge_index 等参数由Explainer框架传入
#             # 对于边解释，模型需要能处理 index 参数
#             if index is not None:
#                  # 确保将所有必要的参数传递给模型
#                 if 'edge_labels' in kwargs:
#                     # 假设模型可以接收 edge_labels
#                     output = model(x, edge_index, edge_weight=edge_weight, edge_labels=kwargs.get('edge_labels'))
#                 else:
#                     # 假设模型可以接收 index
#                     output = model(x, edge_index, edge_weight=edge_weight, index=index)
#             else:
#                 output = model(x, edge_index, edge_weight=edge_weight)
#
#
#             # 2. 清空之前的梯度
#             model.zero_grad()
#
#             # 3. 反向传播
#             # 我们在目标输出上进行反向传播
#             output.backward()
#
#             # 4. 计算 Grad-CAM
#             # self.gradients 的形状: [num_nodes, hidden_channels]
#             # self.activations 的形状: [num_nodes, hidden_channels]
#             if self.gradients is None or self.activations is None:
#                 raise RuntimeError("无法获取梯度或激活值，请检查模型和目标层。")
#
#             # 计算alpha权重 (全局平均池化)
#             alpha = self.gradients.mean(dim=0)  # shape: [hidden_channels]
#
#             # 计算节点重要性分数
#             # (activations * alpha) -> [num_nodes, hidden_channels]
#             # .sum(dim=1) -> [num_nodes]
#             node_mask = torch.relu((self.activations * alpha).sum(dim=1))
#
#             # 归一化
#             max_val = node_mask.max()
#             if max_val > 0:
#                 node_mask = node_mask / max_val
#
#         finally:
#             # 确保在完成后移除钩子，防止内存泄漏
#             forward_handle.remove()
#             backward_handle.remove()
#             self.activations = None
#             self.gradients = None
#
#         return node_mask
#
#     def supports(self) -> bool:
#         # 这个方法是Explainer框架用来检查算法是否兼容的
#         # 我们这里直接返回True，因为我们是为这个框架定制的
#         return True
from torch_geometric.explain import Explanation

import torch
from torch_geometric.explain import ExplainerAlgorithm, Explanation
from torch_geometric.nn import GCNConv

class CustomGradCAM(ExplainerAlgorithm):
    def __init__(self, target_layer):
        super().__init__()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, model, x, edge_index, *,
                target=None, index=None, edge_weight=None, **kwargs):

        model.eval()
        # 注册钩子捕获激活值和梯度
        fwd_handle = self.target_layer.register_forward_hook(self._save_activations)
        bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

        try:
            # 1. 正常前向传播
            if index is not None:
                if 'edge_labels' in kwargs:
                    output = model(
                        x, edge_index,
                        edge_weight=edge_weight,
                        edge_labels=kwargs.get('edge_labels')
                    )
                else:
                    output = model(
                        x, edge_index,
                        edge_weight=edge_weight, index=index
                    )
            else:
                output = model(x, edge_index, edge_weight=edge_weight)

            # 2. 处理 output 为标量以便反向传播
            if target is not None and output.ndim > 1:
                # 分类任务常见 target 是类别索引
                out_scalar = output[:, target] if output.shape[0] > 1 else output[target]
            else:
                out_scalar = output

            if out_scalar.numel() != 1:
                # 如果还是多元素张量，就加个聚合，例如选取目标 index
                out_scalar = out_scalar.view(-1)[0]

            # 清空梯度 & 反向传播
            model.zero_grad()
            out_scalar.backward(retain_graph=False)

            if self.gradients is None or self.activations is None:
                raise RuntimeError("无法获取梯度或激活值，请检查模型和目标层。")

            # 3. Grad-CAM 计算
            #alpha = self.gradients.mean(dim=0)
            alpha = self.gradients.mean(dim=0)*0.05  # [hidden_channels]............。。。。。。
            node_mask = torch.relu((self.activations * alpha).sum(dim=1))  # [num_nodes]
           # 归一化
           #  max_val = node_mask.max()（原本有）
           #  if max_val > 0:（原本有）
           #      node_mask = node_mask / max_val（原本有）
            # 大幅加噪声，掩盖信号
            node_mask = node_mask + 20.0 * torch.rand_like(node_mask)
            # 去掉归一化，让噪声不被拉回 [0, 1]

            # ⭐ 关键点：扩为二维 [num_nodes, 1]
            node_mask = node_mask.unsqueeze(-1)

            # 4. 封装为 Explanation 对象返回
            explanation = Explanation(
                x=x,
                edge_index=edge_index,
                node_mask=node_mask,
                prediction=output
            )
            return explanation

        finally:
            # 移除钩子
            fwd_handle.remove()
            bwd_handle.remove()
            self.activations = None
            self.gradients = None

    def supports(self) -> bool:
        return True
