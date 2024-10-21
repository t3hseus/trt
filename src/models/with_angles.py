import torch
from torch import Tensor, nn


class PointTransformerEncoder(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()

        self.sa1_mh = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)
        self.norm11 = nn.LayerNorm(channels)
        self.ff1 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm12 = nn.LayerNorm(channels)

        self.sa2_mh = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)
        self.norm21 = nn.LayerNorm(channels)
        self.ff2 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm22 = nn.LayerNorm(channels)

        self.sa3_mh = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)
        self.norm31 = nn.LayerNorm(channels)
        self.ff3 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm32 = nn.LayerNorm(channels)

        self.sa4_mh = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)
        self.norm41 = nn.LayerNorm(channels)
        self.ff4 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm42 = nn.LayerNorm(channels)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        batch_size, _, _ = x.size()

        x1, _ = self.sa1_mh(x, x, x, key_padding_mask=~mask)
        x1 = self.norm12(self.ff1(self.norm11(x + x1)))
        x2, _ = self.sa1_mh(x1, x1, x1, key_padding_mask=~mask)
        x2 = self.norm22(self.ff2(self.norm21(x1 + x2)))
        #x3, _ = self.sa2_mh(x2, x2, x2, key_padding_mask=~mask)
        #x3 = self.norm32(self.ff3(self.norm31(x2 + x3)))
        #x4, _ = self.sa3_mh(x3, x3, x3, key_padding_mask=~mask)
        #x4 = self.norm42(self.ff4(self.norm41(x3 + x4)))

        return x2


class TRTDetectDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        channels: int = 128,
        dim_ff: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        return_intermediate: bool = False,
    ) -> None:
        """
        Parameters:
            num_layers: number of decoder blocks aka layers in encoder
            channels: number of input channels, model dimension
            dim_ff: number of channels in the feedforward module in layer.
                channels -> dim_feedforward -> channels
            num_heads: number of attention heads per layer
            dropout: dropout probability
            return_intermediate: if True, intermediate outputs will be
                returned to compute auxiliary losses
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TRTDetectDecoderLayer(
                    channels=channels,
                    dim_ff=dim_ff,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.return_intermediate = return_intermediate
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        query,
        memory,
        memory_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        output = query

        intermediate = []
        for layer in self.layers:
            output = layer(
                query=output,
                memory=memory,
                memory_key_padding_mask=memory_mask,
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
    def __init__(
        self,
        channels: int = 128,
        dim_ff: int = 64,
        num_heads: int = 2,
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
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim_ff, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

        self.dropout = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

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


class TRTHybrid(nn.Module):
    def __init__(
        self,
        num_points: int = 512,
        input_channels: int = 3,
        num_candidates: int = 10,
        num_heads: int = 2,
        num_classes: int = 1,
        num_out_params: int = 7,
        dropout: float = 0.0,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()

        channels = num_points // 4
        self.num_points = num_points
        self.dim_model = channels
        self.num_heads = num_heads
        self.return_intermediate = return_intermediate
        self.emb_encoder = nn.Sequential(
            nn.Linear(input_channels, channels // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channels // 2, channels - 2),
        )

        self.encoder = PointTransformerEncoder(channels=channels - 2)
        self.query_embed = nn.Embedding(
            num_embeddings=num_candidates,
            embedding_dim=channels,
        )
        self.decoder = TRTDetectDecoder(
            channels=channels,
            num_layers=6,
            dim_ff=channels * 2,
            num_heads=2,
            dropout=dropout,
            return_intermediate=return_intermediate,
        )
        self.segmentation_head = nn.Sequential(
            nn.Linear(channels - 2, channels, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channels, channels // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channels // 2, 2),
        )
        self.class_head = nn.Sequential(
            nn.Linear(channels, channels // 2, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channels // 2, num_classes + 1),
        )
        self.params_head = nn.Sequential(
            nn.Linear(channels, channels // 2, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channels // 2, num_out_params - 3),
        )
        self.vertex_head = nn.Sequential(
            nn.Linear(channels, channels // 2),  # num of vertex elements
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channels // 2, 3),  # num of vertex elements
        )

    def forward(
        self, x, mask=None, return_params_with_vertex: bool = False
    ) -> dict[str, Tensor]:
        """
        It returns a dict with the following elements:
        - "pred_logits": the classification logits (including no-object) for all
           queries. Shape= [batch_size x num_queries x (num_classes + 1)]
        - "pred_params": The *normalized* parameters for all queries, represented
           as (x,y,z, pt, phi, theta, charge). These values are normalized in
           [0, 1].
        """
        batch_size = x.shape[0]

        x = self.emb_encoder(x)

        x_encoder = self.encoder(x, mask=mask)
        outputs_segmentation = self.segmentation_head(x_encoder)

        # add segmentation info to encoder
        x_encoder = torch.cat([x_encoder, outputs_segmentation], dim=-1)
        global_feature = x_encoder.mean(dim=-2)
        global_feature = global_feature.squeeze(-2)

        # decoder transformer
        query_pos_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x_decoder = torch.zeros_like(query_pos_embed)
        x = self.decoder(
            memory=x_encoder,
            query=x_decoder,
            query_pos=query_pos_embed,
            memory_mask=mask,
        )

        outputs_class = self.class_head(x)  # no sigmoid, plain logits!
        outputs_params = self.params_head(x)

        # params adaptation
        # outputs_params = torch.zeros(
        #    (*outputs_params_.shape[:-1], outputs_params_.shape[-1] - 1),
        #    dtype=outputs_params_.dtype,
        #    device=outputs_params_.device,
        #)
        #outputs_params[..., 0] = outputs_params_[..., 0]
        #phi_sin = torch.sin(outputs_params_[..., 1])
        phi = torch.acos(torch.tanh(outputs_params[..., 1]))
        outputs_params[..., 1] = (
            # atan2 is in range [-pi; pi], but we expect phi to be from 0 to 2pi and
            # normalize it to lie in range [0; 1]
            (phi + torch.pi) / 2 / torch.pi
        )
        #outputs_params[..., 2] = outputs_params_[..., 3]  # theta
        #outputs_params[..., 3] = outputs_params_[..., 4]  # charge

        outputs_vertex = self.vertex_head(global_feature)

        if return_params_with_vertex:
            # for evaluation (to hide concatenation to
            vertex = outputs_vertex.unsqueeze(-2).expand(
                -1, outputs_params.shape[-2], -1
            )
            if self.return_intermediate:
                vertex = vertex.unsqueeze(0).expand(outputs_params.shape[0], -1, -1, -1)

            outputs_params = torch.cat((vertex, outputs_params), dim=-1)

        return {
            "logits": outputs_class,
            "params": outputs_params,
            "vertex": outputs_vertex,
            "hit_logits": outputs_segmentation,
        }
