from typing import Dict

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch import nn, Tensor

from src.model import PointTransformerEncoder


class TRTBaseline(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        input_channels: int = 3,
        num_heads: int = 4,
        num_candidates: int = 5
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_candidates = num_candidates

        self.activation = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.2)

        self.pre_emb_encoder = nn.Linear(input_channels, channels)

        self.encoder = PointTransformerEncoder(
            channels=channels, num_heads=self.num_heads
        )
        self.segmentation_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            self.activation,
            nn.Linear(channels // 2, channels // 4),
            nn.LayerNorm(channels // 4),
            self.activation,
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x, mask=None) -> Dict[str, Tensor]:
        x_encoder = self.pre_emb_encoder(x)
        x_encoder = self.encoder(x_encoder, mask=mask)
        outputs_segmentation = self.segmentation_head(x_encoder)

        if self.training:
            return {
                "hit_logits": outputs_segmentation,
            }

        mask = (outputs_segmentation.sigmoid() > 0.5).squeeze(-1)
        batch_size = x.shape[0]
        params = []
        cluster_labels = []
        for i in range(batch_size):
            x_hits = x[i, mask[i]].cpu().detach().numpy()

            if x_hits.shape[0] < 10:
                cluster_labels.append(np.zeros(x_hits.shape[0]))
            else:
                # clusters num int(np.round(x_hits.shape[0] / 33))
                clust = DBSCAN(eps=0.1, min_samples=2)
                clust.fit(x_hits)
                cluster_labels.append(clust.labels_)

            params.append([])

        output_params = torch.zeros((len(params), 4))

        return {
            "params": output_params,
            "vertex": torch.tensor(
                [0.5, 0.5, 0.5], dtype=torch.float, device=x.device
            ),
            "hit_logits": outputs_segmentation,
            "cluster_labels": cluster_labels,
        }
