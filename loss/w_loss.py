import torch
import torch.nn as nn
from utils.model_utils import setup_model
import ot

class WLoss(nn.Module):
    def __init__(self, ckpt, device, batch_size):
        super(WLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        inv_net, opts = setup_model(ckpt, device)
        self.generator = inv_net.decoder
        self.e4e = inv_net.encoder
        self.latent_avg = inv_net.latent_avg
        ab_num = batch_size
        self.ab = torch.ones(ab_num) / ab_num
        self.ab = self.ab.to(device)

    def forward(self, inputs, outputs):
        in_codes = self.e4e(inputs)
        in_codes = in_codes + self.latent_avg.repeat(in_codes.shape[0], 1, 1)
        out_codes = self.e4e(outputs)
        out_codes = out_codes + self.latent_avg.repeat(out_codes.shape[0], 1, 1)
        sample_z = torch.randn(self.batch_size, 512, device=self.device)
        real_w = self.generator.get_latent(sample_z)
        real_w = real_w.unsqueeze(1)
        real_w = real_w.repeat(1, 18, 1)
        loss_w_cos = torch.mean(torch.abs(torch.nn.functional.cosine_similarity(in_codes, out_codes, dim=2)))
        ot_distances = []
        for j in range(18):
            M = ot.dist(out_codes[:, j], real_w[:, j])
            ot_distance = ot.emd2(self.ab, self.ab, M) / (5120 * self.batch_size)
            ot_distances.append(ot_distance)
        loss_was = 1 - torch.mean(torch.tensor(ot_distances))
        return loss_w_cos, loss_was
