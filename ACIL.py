import torch
import torch.nn.functional as F
from torch import nn
from AnalyticLinear import RecursiveLinear
from Buffer import RandomBuffer

class ACILClassifierForDETR(nn.Module):
    def __init__(self, input_dim, num_classes, buffer_size=8192, gamma=1e-3, device=None, dtype=torch.double):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

        # 缓存模块：用于特征扩展
        self.buffer = RandomBuffer(input_dim, buffer_size, device=device, dtype=dtype)

        # 解析分类器（RecursiveLinear）
        self.analytic_linear = RecursiveLinear(buffer_size, gamma, device=device, dtype=dtype)

        # eval 模式（解析学习不用 BP）
        self.eval()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        推理逻辑：用于替换 class_embed 的 forward()。
        输入为 Transformer decoder 输出，维度 [B*Q, D]
        """
        X = self.buffer(X)  # 特征扩展
        return self.analytic_linear(X)  # [B*Q, num_classes]

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        用于训练阶段解析更新权重。
        - X: [N, D] decoder 提取的特征（只前景 query）
        - y: [N] long 类型标签
        """
        X = self.buffer(X)  # 特征扩展
        Y = F.one_hot(y, num_classes=self.num_classes).to(X.dtype)  # 变为 float32 或 float64
        self.analytic_linear.fit(X, Y)

    @torch.no_grad()
    def update(self) -> None:
        """
        用于在 batch 式 ACIL 中做最后的权重更新
        """
        self.analytic_linear.update()
