import torch
import torch.nn as nn

class CustomLossFunction(nn.Module):
    def __init__(self, lambda_alpha=1.0):
        super(CustomLossFunction, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_alpha = lambda_alpha

    def forward(self, outputs, labels, alpha):
        loss = self.mse_loss(outputs, labels) + self.lambda_alpha * torch.abs(alpha)
        return loss
