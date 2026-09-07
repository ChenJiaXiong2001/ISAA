"""按通道拓扑细化图卷积核。

该文件对应配置值 `ctr_channel`。实现参考 CTR-GCN 的核心思想：
先根据当前样本的节点特征生成节点间关系，再把关系投影成按输出通道区分的
拓扑修正量。这样不同通道可以关注不同的空间连接模式。
"""

from __future__ import annotations

import torch
from torch import nn

from isaa.models.normalization import MaskedBatchNorm2d, _mask_has_empty_sample


class ChannelWiseTopologyGraphConv(nn.Module):
    """按样本、按通道细化拓扑的 CTR 风格图卷积。

    输入:
        B x C_in x T x N。

    输出:
        B x C_out x T x N。

    计算流程:
        1. 使用 theta/phi 从输入生成 query/key。
        2. 对时间维做 masked mean，得到每个节点的全局关系描述。
        3. 根据 query-key 差值生成 N x N 的动态关系。
        4. 将动态关系投影到 C_out 个通道，并叠加到基础拓扑上。
        5. 对输入特征做 1x1 投影后按动态拓扑聚合。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_adjacency: torch.Tensor,
        *,
        relation_reduction: int = 8,
        min_relation_channels: int = 8,
        topology_scale: float = 1.0,
        diagonal_fast_path: bool = True,
    ) -> None:
        """初始化按通道细化拓扑的图卷积。

        参数:
            relation_reduction: 关系分支的通道压缩比例，数值越大关系分支越轻。
            min_relation_channels: 关系分支保底通道数，避免小模型表达力过弱。
            topology_scale: 动态拓扑修正量的整体缩放系数。
            diagonal_fast_path: 自环图使用等价的逐节点计算，关闭时保留稠密路径。
        """
        super().__init__()
        if base_adjacency.dim() == 3:
            base_adjacency = base_adjacency.sum(dim=0)
        relation_reduction = max(1, int(relation_reduction))
        relation_channels = max(int(out_channels) // relation_reduction, int(min_relation_channels))
        relation_channels = max(1, relation_channels)
        self.register_buffer("base_topology", base_adjacency.t().contiguous())
        self.register_buffer("topology_mask", (base_adjacency.t() > 0).float().contiguous())
        self.topology_scale = float(topology_scale)
        self.theta = nn.Conv2d(in_channels, relation_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, relation_channels, kernel_size=1)
        self.relation_proj = nn.Conv2d(relation_channels, out_channels, kernel_size=1)
        self.feature_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.topology_alpha = nn.Parameter(torch.zeros(1))
        self.proj_norm = MaskedBatchNorm2d(out_channels)
        self.proj_act = nn.ReLU(inplace=True)
        self.diagonal_fast_path = diagonal_fast_path
        self._use_diagonal_topology = self._can_use_diagonal_topology()
        self.register_load_state_dict_post_hook(self._refresh_topology_path)

    def _can_use_diagonal_topology(self) -> bool:
        # Fixed graph buffers are inspected only at construction/checkpoint load,
        # never in the training loop, where this would synchronize CUDA.
        return self.diagonal_fast_path and all(
            bool(torch.count_nonzero(matrix) == torch.count_nonzero(matrix.diagonal()))
            for matrix in (self.base_topology, self.topology_mask)
        )

    def _refresh_topology_path(self, _module: nn.Module, _incompatible_keys: object) -> None:
        self._use_diagonal_topology = self._can_use_diagonal_topology()

    @staticmethod
    def _masked_temporal_mean(x: torch.Tensor, valid_frame_mask: torch.Tensor | None) -> torch.Tensor:
        """按时间维求均值，有 mask 时忽略 padding 帧。

        返回形状为 B x C x N，用于构造节点间关系。该函数被多种 CTR 风格卷积复用。
        """
        if valid_frame_mask is None:
            return x.mean(dim=2)
        b, _, t, _ = x.shape
        mask = valid_frame_mask.to(device=x.device).bool()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(b, -1)
        if mask.shape != (b, t):
            raise ValueError(f"valid_frame_mask 形状必须是 B x T，当前为 {tuple(mask.shape)}，输入为 {tuple(x.shape)}")
        if _mask_has_empty_sample(mask):
            raise ValueError("valid_frame_mask 每个样本至少需要 1 帧有效")
        weight = mask.to(dtype=x.dtype).view(b, 1, t, 1)
        return (x * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)

    def forward(self, x: torch.Tensor, valid_frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """执行按样本、按输出通道生成拓扑细化的图卷积。"""
        query = self._masked_temporal_mean(self.theta(x), valid_frame_mask)
        key = self._masked_temporal_mean(self.phi(x), valid_frame_mask)
        if self._use_diagonal_topology:
            # The 1x1 relation projection does not mix node pairs, so computing
            # only its diagonal preserves the learned self-loop weights.
            relation = torch.tanh(query - key).unsqueeze(-2)
            refinement = self.relation_proj(relation)
            topology = self.base_topology.diagonal().view(1, 1, 1, -1)
            mask = self.topology_mask.diagonal().view(1, 1, 1, -1)
            topology = topology + self.topology_alpha * self.topology_scale * refinement * mask
            x = self.feature_proj(x)
            # Dense einsum autocasts its operands; elementwise multiplication
            # does not. Match that cast to retain the AMP output dtype.
            x = topology.to(dtype=x.dtype) * x
        else:
            relation = torch.tanh(query.unsqueeze(-1) - key.unsqueeze(-2))
            refinement = self.relation_proj(relation)
            topology = self.base_topology.view(1, 1, *self.base_topology.shape)
            mask = self.topology_mask.view(1, 1, *self.topology_mask.shape)
            topology = topology + self.topology_alpha * self.topology_scale * refinement * mask
            x = self.feature_proj(x)
            x = torch.einsum("bcuv,bctv->bctu", topology, x)
        x = self.proj_norm(x, valid_frame_mask=valid_frame_mask)
        return self.proj_act(x)
