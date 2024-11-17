from abc import ABC, abstractmethod

from torch import Tensor, nn

from src.models.layers import TRTEncoderLayer


class BaseEncoder(nn.Module):
    """Base class for encoders"""

    def __init__(
        self,
        channels: int = 64,
    ):
        self.channels = channels
        super().__init__()

    def forward(self, x: Tensor, mask=None) -> Tensor:
        raise NotImplementedError


class PointTransformerEncoder(BaseEncoder):
    def __init__(
        self,
        channels: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.activation = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.2)

        self.sa1_mh = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.norm11 = nn.LayerNorm(channels)
        self.ff1 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            self.activation,
            nn.Linear(channels * 2, channels),
        )
        self.norm12 = nn.LayerNorm(channels)

        self.sa2_mh = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.norm21 = nn.LayerNorm(channels)
        self.ff2 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            self.activation,
            nn.Linear(channels * 2, channels),
        )
        self.norm22 = nn.LayerNorm(channels)

        self.sa3_mh = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.norm31 = nn.LayerNorm(channels)
        self.ff3 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            self.activation,
            nn.Linear(channels * 2, channels),
        )
        self.norm32 = nn.LayerNorm(channels)

        self.sa4_mh = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.norm41 = nn.LayerNorm(channels)
        self.ff4 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            self.activation,
            nn.Linear(channels * 2, channels),
        )
        self.norm42 = nn.LayerNorm(channels)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        batch_size, _, _ = x.size()

        x1, _ = self.sa1_mh(x, x, x, key_padding_mask=~mask)
        x1 = self.norm12(self.ff1(self.norm11(x + x1)))
        x2, _ = self.sa2_mh(x1, x1, x1, key_padding_mask=~mask)
        x2 = self.norm22(self.ff2(self.norm21(x1 + x2)))
        x3, _ = self.sa3_mh(x2, x2, x2, key_padding_mask=~mask)
        x3 = self.norm32(self.ff3(self.norm31(x2 + x3)))
        x4, _ = self.sa4_mh(x3, x3, x3, key_padding_mask=~mask)
        x4 = self.norm42(self.ff4(self.norm41(x3 + x4)))

        return x4


class TRTEncoder(BaseEncoder):
    def __init__(
        self,
        channels: int = 64,
        dim_ff: int = 128,
        dropout: float = 0.1,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TRTEncoderLayer(
                    channels=channels,
                    dim_ff=dim_ff,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, mask=None) -> Tensor:
        output = x
        for layer in self.layers:
            output = layer(x=output, mask=mask)
        return output


class HitEncoder(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        activation: nn.Module,
        in_channels: int = 3,
        channels: int = 64,
    ) -> None:
        super().__init__()
        self.pre_emb_encoder = nn.Sequential(
            nn.Linear(in_channels, channels * 2),
            activation,
            nn.Linear(channels * 2, channels),
        )
        self.encoder = encoder

    def forward(self, x: Tensor, mask=None) -> Tensor:
        x = self.pre_emb_encoder(x)
        return self.encoder(x, mask=mask)
