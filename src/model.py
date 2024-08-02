import gin
import torch
import torch.nn as nn
import torch.nn.functional as f


class PointTransformerEncoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        # channels = int(n_points / 4)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(channels * 4, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.sa1 = SALayer(channels)
        self.sa2 = SALayer(channels)
        self.sa3 = SALayer(channels)
        self.sa4 = SALayer(channels)

    def forward(self, x, mask=None):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, _ = x.size()

        # B, D, N
        # x = f.relu(self.bn1(self.conv1(x)))
        # x = f.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x, mask=mask)
        x2 = self.sa2(x1, mask=mask)
        x3 = self.sa3(x2, mask=mask)
        x4 = self.sa4(x3, mask=mask)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv3(x)
        return x


class SALayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)  # why not layer norm?
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)  # why not normalization?
        d = energy.shape[-2]
        if mask is not None:
            mask = torch.bitwise_and(mask[:, :, None].bool(), mask[:, None, :].bool())
            energy = energy.masked_fill(mask == 0, -9e15)
            d = 1e-9 + mask.sum(dim=1, keepdim=True)
        energy = energy / d
        attention = self.softmax(energy)  # need normalization!!?
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

    @staticmethod
    def _generate_square_subsequent_mask(sequence_mask):
        mask = (torch.triu(torch.ones(sequence_mask, sequence_mask)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class JointAttentionLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)  # why not layer norm?
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None):
        # b, n, c
        x_q = self.q_conv(query).permute(0, 2, 1)  # do we need permute or not?
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)  # why not normalization?
        d = energy.shape[-2]
        if mask is not None:
            mask = torch.bitwise_and(mask[:, :, None].bool(), mask[:, None, :].bool())
            energy = energy.masked_fill(mask == 0, -9e15)
            d = 1e-9 + mask.sum(dim=1, keepdim=True)
        energy = energy / d
        attention = self.softmax(energy)  # need normalization!!?
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

    @staticmethod
    def _generate_square_subsequent_mask(sequence_mask):
        mask = (torch.triu(torch.ones(sequence_mask, sequence_mask)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class PCTDetectDecoder(nn.Module):
    def __init__(self, channels=128, dim_feedforward=64, nhead=2, dropout=0.2):
        super().__init__()

        self.sa1 = nn.MultiheadAttention(
            channels, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            channels, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = torch.nn.functional.leaky_relu  # gelu, leaky_relu etc

    def forward(
        self,
        x_encoder: torch.Tensor,
        x_decoder: torch.Tensor,
        object_pos_emb: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        # permute reshape
        batch_size, _, _ = x_encoder.size()
        x_encoder = x_encoder.permute(0, 2, 1)
        # B, D, N
        # self-attention, add + norm for query_embeddings
        q = k = x_decoder + object_pos_emb
        mask = ~mask
        x_att = self.sa1(q, k, value=x_decoder)[0]
        x1 = self.norm1(x_decoder + self.dropout1(x_att))

        # combine with encoder output! (attention + add+norm
        x_att = self.multihead_attn(
            query=(x1 + object_pos_emb),
            key=x_encoder,
            value=x_encoder,
            key_padding_mask=mask,
        )[0]
        x1 = self.norm2(x1 + self.dropout2(x_att))
        # fpn on top of layer
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x1))))
        x1 = x1 + self.dropout3(x2)
        x1 = self.norm3(x1)
        return x1


class TRT(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        n_points: int = 512,
        input_channels: int = 3,
        initial_permute: bool = True,
        num_candidates: int = 10,
        nhead: int = 2,
        num_classes: int = 1,
        num_out_params: int = 7,
    ):
        super().__init__()
        channels = n_points // 4
        self.n_points = n_points
        self.initial_permute = initial_permute
        self.d_model = channels
        self.nhead = nhead
        self.emb_encoder = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.encoder = PointTransformerEncoder(channels=channels)

        self.query_embed = nn.Embedding(
            num_embeddings=num_candidates, embedding_dim=channels
        )
        self.decoder = PCTDetectDecoder(
            channels=channels, dim_feedforward=channels // 2, nhead=2, dropout=dropout
        )
        self.class_head = nn.Linear(channels, num_classes + 1)

        self.params_head = nn.Sequential(
            nn.Linear(channels, num_out_params * 2, bias=False),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(num_out_params * 2, num_out_params, bias=False),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, inputs, mask=None) -> dict[str, torch.Tensor]:
        """
        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all
                            queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_params": The *normalized* parameters for all queries, represented
                            as (x,y,z, pt, phi, theta, charge).
                           These values are normalized in [0, 1].
        """
        if self.initial_permute:
            inputs = inputs.permute(0, 2, 1)
        batch_size, _, n = inputs.size()  # B, D, N

        x = self.emb_encoder(inputs)
        x_encoder = self.encoder(x, mask=mask)
        # decoder transformer
        query_pos_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x_decoder = torch.zeros_like(query_pos_embed)

        x = self.decoder(x_encoder, x_decoder, query_pos_embed, mask=mask)
        outputs_class = self.class_head(x)  # no sigmoid, plain logits!
        outputs_coord = self.params_head(
            x
        ).sigmoid()  # params are normalized after sigmoid!!
        return {
            "logits": outputs_class,
            "params": outputs_coord,
        }


if __name__ == "__main__":
    model = TRT()
