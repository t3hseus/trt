import copy

import gin
import torch
import torch.nn as nn
import torch.nn.functional as f


class TRTEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 512,
        nhead: int = 4,
        num_layers=4,
        dropout: float = 0.2,
        dim_feedforward=64,
    ):
        super().__init__()
        module = TRTEncoderLayer(
            channels=channels,
            dim_feedforward=dim_feedforward,
            n_heads=nhead,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)

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
        # x1 = self.sa1(x, mask=mask)
        # x2 = self.sa2(x1, mask=mask)
        # x3 = self.sa3(x2, mask=mask)
        # x4 = self.sa4(x3, mask=mask)
        output = x

        for layer in self.layers:
            output = layer(output, mask=mask, key_padding_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        # x = torch.cat((x1, x2, x3, x4), dim=1)
        # x = self.conv3(x)
        return x


class TRTEncoderLayer(nn.Module):
    def __init__(
        self,
        channels,
        n_heads,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            channels, n_heads, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = torch.nn.functional.leaky_relu

    def forward(
        self,
        x,
        mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ):
        q = k = x
        x2 = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x)
        x = self.norm2(x)
        return x


class TRTDetectDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        channels: int = 128,
        dim_feedforward: int = 64,
        nhead: int = 4,
        dropout: float = 0.2,
        return_intermediate: bool = False,
    ):
        """
        Parameters:
            num_layers: number of decoder blocks aka layers in encoder
            channels: number of input channels, model dimension
            dim_feedforward: number of channels in the feedforward module in layer.
                channels -> dim_feedforward -> channels
            nhead: number of attention heads per layer
            dropout: dropout probability
            return_intermediate: if True, intermediate outputs will be
                returned to compute auxiliary losses
        """
        super().__init__()
        module = TRTDetectDecoderLayer(
            channels=channels,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])
        self.return_intermediate = return_intermediate
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        query,
        memory,
        permute_input: bool = True,
        query_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        query_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = query
        if permute_input:
            # permute reshape
            output = output.permute(0, 2, 1)
            # B, D, N
        intermediate = []
        for layer in self.layers:
            output = layer(
                query=output,
                memory=memory,
                query_mask=query_mask,
                # memory_mask=memory_mask,
                # query_key_padding_mask=query_key_padding_mask,
                memory_key_padding_mask=memory_mask,
                # memory_pos=memory_pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)
            return torch.stack(intermediate)

        return output


class TRTDetectDecoderLayer(nn.Module):
    def __init__(self, channels=128, dim_feedforward=64, nhead=2, dropout=0.2):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            channels, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
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
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor = None,
        query_mask: torch.Tensor | None = None,
    ):

        # self-attention, add + norm for query_embeddings
        q = k = query + query_pos
        # mask = ~query_mask
        x_att = self.self_attn(q, k, value=query)[0]
        query = self.norm1(query + self.dropout1(x_att))

        # combine with encoder output! (attention + add+norm
        x_att = self.cross_attn(
            query=(query + query_pos),
            key=memory,
            value=memory,
            key_padding_mask=~memory_key_padding_mask,
            attn_mask=memory_mask,
        )[0]
        x = self.norm2(query + self.dropout2(x_att))
        # fpn on top of layer
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x


class TRT(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        n_points: int = 512,
        input_channels: int = 3,
        initial_permute: bool = True,
        num_candidates: int = 10,
        nhead: int = 2,
        num_charges: int = 2,
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

        self.query_embed = nn.Embedding(
            num_embeddings=num_candidates, embedding_dim=channels
        )

        self.encoder = TRTEncoder(
            channels=channels, dim_feedforward=channels * 2, nhead=2, dropout=dropout
        )
        self.decoder = TRTDetectDecoder(
            channels=channels, dim_feedforward=channels * 2, nhead=2, dropout=dropout
        )

        self.class_head = nn.Sequential(nn.Linear(channels, num_charges + 1))

        self.params_head = nn.Sequential(
            nn.Linear(channels, num_out_params * 2, bias=False),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(num_out_params * 2, num_out_params, bias=False),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, inputs, mask=None, orig_params=None) -> dict[str, torch.Tensor]:
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
        batch_size, dim, n = inputs.size()  # B, D, N

        x_emb = self.emb_encoder(inputs)
        x_emb = x_emb.permute(0, 2, 1)
        x_encoder = self.encoder(x_emb, mask=mask)
        # decoder transformer
        query_pos_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x_decoder = torch.zeros_like(query_pos_embed)  # можно сделать linear(memory)

        x = self.decoder(
            query=x_decoder,
            memory=x_encoder,
            permute_input=False,
            query_pos=query_pos_embed,
            memory_mask=mask,
        )
        outputs_class = self.class_head(
            x
        )  # no sigmoid, plain logits! output is [+1, -1, no_object]
        outputs_coord = self.params_head(
            x
        ).sigmoid()  # params are normalized after sigmoid!!
        return {
            "logits": outputs_class,
            "params": outputs_coord,
        }


if __name__ == "__main__":
    model = TRT()
