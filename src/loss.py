# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def mask_correlated_samples(batch_size: int) -> torch.Tensor:
        n = 2 * batch_size
        mask = torch.ones((n, n), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        n = 2 * batch_size
        device = z_i.device

        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature

        mask = self.mask_correlated_samples(batch_size).to(device)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n, 1)
        negative_samples = sim[mask].reshape(n, -1)
        labels = torch.zeros(n, device=device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= n
        return loss
