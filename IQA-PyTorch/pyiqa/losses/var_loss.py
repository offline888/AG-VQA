import torch
from torch import nn
from pyiqa.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class VRLoss(nn.Module):
    """
    方差正则化 Loss (Variance Regularization Loss)
    用于约束概率模型（如 PEVQA）中的方差，防止其退化为 0 或爆炸。
    """
    def __init__(self, loss_weight=1e-4, target_var=1.0):
        super().__init__()
        # 把 YML 里传进来的权重和超参数存下来
        self.loss_weight = loss_weight
        self.target_var = target_var

    def forward(self, video_var, text_var):
        """
        计算方差正则惩罚
        Args:
            video_var (Tensor): 视频端预测的方差 [B, D]
            text_var (Tensor): 文本端预测的方差 [5, D]
        """
        l_video = torch.mean((video_var - self.target_var)**2)
        l_text = torch.mean((text_var - self.target_var)**2)
        
        return self.loss_weight * (l_video + l_text)