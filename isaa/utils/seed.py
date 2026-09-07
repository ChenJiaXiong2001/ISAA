"""随机种子工具。

文档小结对齐：
- 这里负责 dummy 验证与小样本训练路径的随机源固定。
- 真实训练还需要固定数据划分、增强参数和原型初始化记录。
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """固定常用随机源，便于复现实验。

    参数:
        seed: 随机种子。

    覆盖范围:
        Python random、NumPy、torch CPU、torch CUDA。

    注意:
        这里没有强制设置 torch.backends.cudnn.deterministic，
        因为当前目标是 smoke test，可按真实复现实验需求后续扩展。
    """
    # 技术备注：复现实验时还要固定数据划分、增强参数和原型初始化方式。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
