import torch
import torch.nn as nn


class DAFN(nn.Module):
    """
    Demographic-Aware Feature Normalization:
    Computes EMA-based demographic statistics and normalizes text embeddings.
    """

    def __init__(self, embed_dim, momentum=0.1, num_groups=4):
        super().__init__()
        self.momentum = momentum
        self.embed_dim = embed_dim
        self.num_groups = num_groups

        self.register_buffer("ema_mean", torch.zeros(num_groups, embed_dim))
        self.register_buffer("ema_std", torch.ones(num_groups, embed_dim))

    def update_stats(self, embeds, demographic):
        for g in range(self.num_groups):
            idx = demographic[:, g] > 0
            if idx.sum() == 0:
                continue

            group_emb = embeds[idx]
            mean = group_emb.mean(dim=0)
            std = group_emb.std(dim=0)

            self.ema_mean[g] = (
                self.momentum * mean + (1 - self.momentum) * self.ema_mean[g]
            )
            self.ema_std[g] = (
                self.momentum * std + (1 - self.momentum) * self.ema_std[g]
            )

    def forward(self, embeds, demographic):
        with torch.no_grad():
            self.update_stats(embeds, demographic)

        # weighted group normalization
        weight = demographic.unsqueeze(-1)
        mu = (weight * self.ema_mean).sum(dim=1)
        sigma = (weight * self.ema_std).sum(dim=1)

        return (embeds - mu) / (sigma + 1e-6)
