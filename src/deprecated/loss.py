# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn

from src.deprecated.matcher import HungarianMatcher
from src.postprocess import TorchTrackGenerator, EventRecoveryFromPredictions


class MatchingLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"

    Args:
        matcher (`src.matcher.HungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        event_generator: EventRecoveryFromPredictions,
        hits_generator: TorchTrackGenerator,
        num_classes: int,
        eos_coef: float,
        losses: list[str],
        charge_as_class: bool = False,
        weights_dict: DictConfig | None = None,  # this is hack
    ):
        super().__init__()
        self.matcher = matcher
        self.charge_as_class = charge_as_class
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        self.torch_hits_generator = hits_generator
        # We need to get one class based on torch for convenience (i think) to get loss
        # and dataset
        self.event_generator = event_generator # For visualization purposes
        # place buffer to the appropiate device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        empty_weight = torch.ones(self.num_classes + 1, device=device)
        empty_weight[-1] = self.eos_coef
        if not weights_dict:
            self.weight_dict = {"loss_ce": 1.0, "loss_params": 1.0, "loss_hits": 0.001}
        else:
            weight_dict = OmegaConf.to_container(weights_dict, resolve=True)
            self.weight_dict = {k: float(v) for k, v in weight_dict.items()}
        self.register_buffer("empty_weight", empty_weight)

    @property
    def __name__(self):
        return str(self.__class__.__name__) + "_" + "_".join(self.losses)

    def loss_labels(self, outputs, targets, indices, num_tracks):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels"
        containing a tensor of dim  [nb_target_tracks]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]
        idx = self._get_source_permutation_idx(indices)
        # target_classes_o = #[torch.ones(len(J)) for t, (_, J) in zip(targets, indices)]
        target_classes_o = torch.cat(
            [t["class_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes_o = target_classes_o.to(torch.int64)
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=source_logits.device,
        )
        target_classes[idx] = target_classes_o
        loss_ce = nn.functional.cross_entropy(
            source_logits.transpose(1, 2), target_classes, self.empty_weight
        )

        losses = {"loss_ce": loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_tracks):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted tracks with real tracks.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor(
            [len(v["class_labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_params(self, outputs, targets, indices, num_tracks):
        """
        Compute the losses related to the parameters estimation and the L1 regression loss

        Targets dicts must contain the key "params" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "params" not in outputs:
            raise KeyError("No predicted parameters found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_params = outputs["params"][idx]
        target_params = torch.cat(
            [t["params"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_params = nn.functional.mse_loss(
            source_params.to(torch.float),
            target_params.to(torch.float),
            reduction="none",
        )
        losses = {}
        losses["loss_params"] = loss_params.sum() / num_tracks
        return losses

    def loss_hits(self, outputs, targets, indices, num_params):
        ## TODO: change numpy hits generation to torch generation
        idx = self._get_source_permutation_idx(indices)
        source_params = outputs["params"][idx][..., :-1]
        source_charges = torch.argmax(outputs["logits"][idx], dim=-1).to(torch.float)
        target_params = torch.cat(
            [t["params"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        source_charges = (outputs["params"][idx][..., -1] > 0.5).to(torch.int32)
        # is_object = source_charges < 2

        # if not (is_object.any()):
        #    return {"loss_hits": 100.}
        is_object = torch.ones_like(source_charges, dtype=torch.bool)
        # source_charges = source_charges[is_object > 0] * 2 - 1
        source_charges = source_params[..., -1] * 2 - 1

        source_charges = source_charges.unsqueeze(-1)
        source_params = torch.concat((source_params[is_object], source_charges), dim=-1)

        source_tracks, _ = self.torch_hits_generator.generate_tracks(
            source_params  # .detach().cpu().numpy()
        )
        # target_charges = torch.cat(
        #    [t["class_labels"][i] for t, (_, i) in zip(targets, indices)], dim=0
        # )
        #unnormalize charge
        target_charges = target_params[..., -1]
        target_charges = target_charges.to(torch.float) * 2 - 1
        target_charges = target_charges.unsqueeze(-1)
        target_params = torch.concat((target_params[..., :-1], target_charges), dim=-1)[
            is_object
        ]

        target_tracks, _ = self.torch_hits_generator.generate_tracks(target_params)
        dists = torch.tensor(0.0)
        if not (len(source_tracks)):
            return torch.tensor(1000.0)
        num_tracks = 0
        for pred_track, target_track in zip(
            source_tracks, target_tracks
        ):
            # it is crucial to use paddings in case of tracks with different lengths
            n_stations = max(pred_track.shape[0], target_track.shape[0])
            target_track_padded = torch.zeros(
                (n_stations, target_track.shape[1]), dtype=torch.float32
            )
            target_track_padded[: len(target_track)] = target_track
            pred_track_padded = torch.zeros(
                (n_stations, pred_track.shape[1]), dtype=torch.float32
            )
            pred_track_padded[: len(pred_track)] = pred_track
            # calculate dists
            # TODO: calculate mean instead of max???
            distance = self._dist(pred_track_padded, target_track_padded).max()
            if ~torch.isnan(distance):
                dists += distance
                num_tracks += 1
        return {"loss_hits": dists / (num_tracks + 1)}

    def _dist(self, x_1, x_2):
        return torch.sqrt(
            (x_2[:, 0] - x_1[:, 0]) ** 2
            + (x_2[:, 1] - x_1[:, 1]) ** 2
            + (x_2[:, 2] - x_1[:, 2]) ** 2
        )

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(source, i) for i, (source, _) in enumerate(indices)]
        )
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(target, i) for i, (_, target) in enumerate(indices)]
        )
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(
        self,
        loss_name: str,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        indices: torch.Tensor,
        num_boxes: int,
    ):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "params": self.loss_params,
            "hits": self.loss_hits,
        }
        if loss_name not in loss_map:
            raise ValueError(f"Loss {loss_name} not supported")
        return loss_map[loss_name](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "auxiliary_outputs"
        }
        # Retrieve the matching between the outputs of the last layer and the targets
        targets_dict = [
            {
                "params": targets[idx],
                "class_labels": torch.zeros(len(targets[idx])),  # charge
            }
            for idx in range(targets.shape[0])
        ]
        # match param vectors
        # TODO: why without charge???
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target tracks across all nodes, for normalization purposes
        num_tracks = targets.shape[0] * targets.shape[1]
        num_tracks = torch.as_tensor(
            [num_tracks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        world_size = 1
        num_tracks = torch.clamp(num_tracks / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets_dict, indices, num_tracks)
            )

        weight_dict = self.weight_dict
        loss = sum(
            losses[k] * weight_dict[k] for k in losses.keys() if k in self.weight_dict
        )

        return loss
