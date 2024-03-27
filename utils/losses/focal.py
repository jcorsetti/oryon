import torch
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(torch.nn.Module):
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    def __init__(self, alpha: float=0.25, gamma:float=2.):

        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor) -> float:

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

