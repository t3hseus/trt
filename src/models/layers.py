from torch import Tensor, nn


class TRTDetectDecoderLayer(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        dim_ff: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )

        self.lin1 = nn.Linear(channels, dim_ff)
        self.lin2 = nn.Linear(dim_ff, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        query_pos: Tensor,
        memory_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor = None,
    ) -> Tensor:
        q = k = query + query_pos
        x_att = self.self_attn(q, k, value=query)[0]
        query = self.norm1(query + self.dropout(x_att))
        x_att = self.cross_attn(
            query=(query + query_pos),
            key=memory,
            value=memory,
            key_padding_mask=~memory_key_padding_mask,
            attn_mask=memory_mask,
        )[0]
        x = self.norm2(query + self.dropout(x_att))
        x2 = self.lin2(self.dropout(self.activation(self.lin1(x))))
        x = x + self.dropout(x2)
        x = self.norm3(x)
        return x


class TRTEncoderLayer(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        dim_ff: int = 32,
        num_heads: int = 4,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(channels, dim_ff),
            activation,
            nn.Linear(dim_ff, channels),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        x1, _ = self.self_attn(x, x, x, key_padding_mask=~mask)
        x1 = self.norm2(self.ff(self.norm1(x + x1)))
        return x1
