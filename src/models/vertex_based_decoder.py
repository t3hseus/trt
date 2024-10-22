"""
This is the best model on 22.10 (morning)
"""
import torch
from torch import Tensor, nn


class PointTransformerEncoder(nn.Module):
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


class TRTDetectDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        channels: int = 64,
        dim_ff: int = 128,
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


class TRTHybrid(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        num_points: int = 512,
        num_candidates: int = 10,
        input_channels: int = 3,
        num_heads: int = 4,
        num_classes: int = 1,
        num_out_params: int = 7,
        num_detector_layers: int = 4,
        dropout: float = 0.0,
        return_intermediate: bool = False,
        zero_based_decoder: bool = True
    ) -> None:
        super().__init__()

        # channels = num_points // 4
        self.num_points = num_points
        self.dim_model = channels
        self.num_heads = num_heads
        self.return_intermediate = return_intermediate
        self.num_candidates = num_candidates
        self.zero_based_decoder = zero_based_decoder

        self.activation = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.2)

        self.pre_emb_encoder = nn.Sequential(
            nn.Linear(input_channels, channels * 2),
            self.activation,
            nn.Linear(channels * 2, channels),
        )
        self.encoder = PointTransformerEncoder(
            channels=channels, num_heads=self.num_heads
        )
        self.post_emb_encoder = nn.Sequential(
            nn.Linear(channels + 2, channels),
            self.activation,
            nn.Linear(channels, channels),
        )

        self.query_embed = nn.Embedding(
            num_embeddings=num_candidates, embedding_dim=channels
        )
        self.decoder = TRTDetectDecoder(
            channels=channels,
            num_layers=num_detector_layers,
            dim_ff=channels * 2,
            num_heads=self.num_heads,
            dropout=dropout,
            return_intermediate=return_intermediate,
        )

        self.segmentation_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            self.activation,
            nn.Linear(channels // 2, channels // 4),
            nn.LayerNorm(channels // 4),
            self.activation,
            nn.Linear(channels // 4, 2),
        )
        self.class_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            self.activation,
            nn.Linear(channels // 2, channels // 4),
            nn.LayerNorm(channels // 4),
            self.activation,
            nn.Linear(channels // 4, num_classes + 1),
        )
        self.params_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            self.activation,
            nn.Linear(channels // 2, channels // 4),
            nn.LayerNorm(channels // 4),
            self.activation,
            nn.Linear(channels // 4, num_out_params - 3),
        )
        self.vertex_head = nn.Sequential(
            nn.Linear(channels, channels // 2),  # num of vertex elements
            nn.LayerNorm(channels // 2),
            self.activation,
            nn.Linear(channels // 2, channels // 4),
            nn.LayerNorm(channels // 4),
            self.activation,
            nn.Linear(channels // 4, 3),  # num of vertex elements
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

        x = self.pre_emb_encoder(x)
        x_encoder = self.encoder(x, mask=mask)
        outputs_segmentation = self.segmentation_head(x_encoder)

        # add segmentation info to encoder
        x_encoder = torch.cat([x_encoder, outputs_segmentation], dim=-1)
        x_encoder = self.post_emb_encoder(x_encoder)

        # as soft mask (if use >, then the result may be 0 (no signal at all)
        seg_mask = torch.softmax(outputs_segmentation, dim=-1)[:, :, 1]
        denom = torch.sum(seg_mask, -1, keepdim=True) + 0.1
        global_feature = torch.sum(x_encoder * mask.unsqueeze(-1), dim=1) / denom
        # global_feature = x_encoder.mean(dim=-2)
        if global_feature.shape[0] > 1:
            global_feature = global_feature.squeeze(-2)  # If we have 1-el batch (for test and for simple train)
        outputs_vertex = self.vertex_head(global_feature)

        # decoder transformer
        query_pos_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.zero_based_decoder:
            x_decoder = torch.zeros_like(query_pos_embed)
        else:
            x_decoder = global_feature.unsqueeze(1).repeat(1, self.num_candidates, 1)
        x = self.decoder(
            memory=x_encoder,
            query=x_decoder,
            query_pos=query_pos_embed,
            memory_mask=mask,
        )
        outputs_class = self.class_head(x)  # no sigmoid, plain logits!
        outputs_coord = self.params_head(x)

        if return_params_with_vertex:
            # for evaluation (to hide concatenation to
            vertex = outputs_vertex.unsqueeze(-2).expand(
                -1, outputs_coord.shape[-2], -1
            )
            if self.return_intermediate:
                vertex = vertex.unsqueeze(0).expand(outputs_coord.shape[0], -1, -1, -1)

            outputs_coord = torch.cat((vertex, outputs_coord), dim=-1)

        return {
            "logits": outputs_class,
            "params": outputs_coord,
            "vertex": outputs_vertex,
            "hit_logits": outputs_segmentation,
        }
