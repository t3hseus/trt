import torch
from torch import Tensor, nn

from src.models.layers import TRTDetectDecoderLayer


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


class TRTHybrid(nn.Module):
    def __init__(
        self,
        hit_encoder: nn.Module,
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
        zero_based_decoder: bool = False,
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
        self.encoder = hit_encoder
        self.post_emb_encoder = nn.Sequential(
            nn.Linear(channels + 1, channels),
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
            nn.Linear(channels // 4, 1),
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

        self.query_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            self.activation,
        )
        self.params_head = nn.Sequential(
            nn.Linear(channels // 2, channels // 4),
            nn.LayerNorm(channels // 4),
            self.activation,
            nn.Linear(channels // 4, num_out_params - 3),
        )
        self.coords_head = nn.Sequential(
            nn.Linear(channels // 2, channels // 4),
            nn.LayerNorm(channels // 4),
            self.activation,
            nn.Linear(channels // 4, 6),
        )
        self.vertex_head = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.LayerNorm(channels // 4),
            self.activation,
            nn.Linear(channels // 4, 3),  # num of vertex elements
        )
        self.queries_init_layer = nn.Linear(1, self.num_candidates)

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
        x_encoder = self.encoder(x, mask=mask)
        outputs_segmentation = self.segmentation_head(x_encoder)

        # add segmentation info to encoder
        x_encoder = torch.cat([x_encoder, outputs_segmentation], dim=-1)
        x_encoder = self.post_emb_encoder(x_encoder)

        # as soft mask (if use >, then the result may be 0 (no signal at all)
        seg_mask = outputs_segmentation.sigmoid()
        denom = torch.sum(seg_mask, 1) + 0.1
        global_feature = torch.sum(x_encoder * mask.unsqueeze(-1), dim=1) / denom

        # global_feature = x_encoder.mean(dim=-2)
        if global_feature.shape[0] > 1:
            # If we have 1-el batch (for test and for simple train)
            global_feature = global_feature.squeeze(-2)
        outputs_vertex = self.vertex_head(global_feature)

        # decoder transformer
        query_pos_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.zero_based_decoder:
            x_decoder = torch.zeros_like(query_pos_embed)
        else:
            x_decoder = self.queries_init_layer(global_feature.unsqueeze(-1)).permute(
                0, 2, 1
            )
            # x_decoder = global_feature.repeat(1, self.num_candidates, 1)
        x = self.decoder(
            memory=x_encoder,
            query=x_decoder,
            query_pos=query_pos_embed,
            memory_mask=mask,
        )
        outputs_class = self.class_head(x)  # no sigmoid, plain logits!
        x = self.query_head(x)
        outputs_params = self.params_head(x)
        outputs_coord = self.coords_head(x)
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
            "params": outputs_params,
            "coords": outputs_coord,
            "vertex": outputs_vertex,
            "hit_logits": outputs_segmentation,
        }
