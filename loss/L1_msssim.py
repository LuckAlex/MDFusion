import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ms_ssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, output, target):
        l1 = self.l1_loss(output, target)
        ms_ssim = self.ms_ssim_loss(output, target)

        combined_loss = self.alpha * l1 + (1 - self.alpha) * (1 - ms_ssim)
        return combined_loss


