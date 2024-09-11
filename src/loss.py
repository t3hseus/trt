# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.postprocess import EventRecoveryFromPredictions


# taken from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        params_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
    """

    def __init__(self, class_cost: float = 1, params_cost: float = 1):
        super().__init__()
        self.class_cost = class_cost
        self.params_cost = params_cost
        if class_cost == 0 and params_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets) -> list[tuple[torch.Tensor]]:
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes]
                with the classification logits
                * "params": Tensor of dim [batch_size, num_queries, 6] with
                the predicted track parameters.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target
                is a dict containing:
                * "class_labels": Tensor of dim [num_target_tracks]
                 (where num_target_tracks is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "params": Tensor of dim [num_target_tracks, 6] containing
                the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of
            (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(
            num_queries, num_target_tracks)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_param = outputs["params"].flatten(
            0, 1)  # [batch_size * num_queries, 7]

        # Also concat the target labels and boxes
        # target_ids = torch.cat([v["class_labels"] for v in targets])
        # dummy targets
        target_ids = torch.ones(
            targets.shape[0] * targets.shape[1], dtype=torch.int)
        target_param = targets.flatten(
            0, 1
        )  # torch.cat([v["params"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between parameters
        params_cost = torch.cdist(out_param.to(
            torch.float32), target_param.to(torch.float32), p=1)

        # Final cost matrix
        cost_matrix = self.params_cost * params_cost + self.class_cost * class_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [
            targets.shape[1] for _ in range(targets.shape[0])
        ]  # [len(v["params"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(cost_matrix.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


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
        matcher (`HungarianMatcher`):
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
        hits_generator: EventRecoveryFromPredictions,
        num_classes: int,
        eos_coef: float,
        losses: list[str],
    ):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        self.hits_generator = hits_generator

        # place buffer to the appropiate device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        empty_weight = torch.ones(self.num_classes + 1, device=device)
        empty_weight[-1] = self.eos_coef
        self.weight_dict = {"loss_ce": 0.5,
                            "loss_params": 1.0, "loss_hits": 0.001}
        self.register_buffer("empty_weight", empty_weight)

    @property
    def __name__(self):
        return str(self.__class__.__name__) + "_" + "_".join(self.losses)

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
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
        card_err = nn.functional.l1_loss(
            card_pred.float(), target_lengths.float())
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
        idx = self._get_source_permutation_idx(indices)
        predicted_tracks = self.hits_generator(
            pred_params=outputs["params"][idx],
            pred_charges=outputs["logits"][idx],
            group_by_tracks=True
        )

        target_tracks = self.hits_generator(
            # TODO: rewrite to omit such weird indexing
            pred_params=torch.cat(
                [t["params"][i] for t, (_, i) in zip(targets, indices)], dim=0
            ),
            pred_charges=torch.cat(
                [t["class_labels"][i] for t, (_, i) in zip(targets, indices)], dim=0
            ),
            from_targets=True,
            group_by_tracks=True
        )

        dists = torch.tensor(0.0)
        if not (len(predicted_tracks)):
            return torch.tensor(1000.0)

        for pred_track, target_track in zip(predicted_tracks, target_tracks):
            if len(target_track) == 0:
                dists += torch.tensor(100.0)
                continue
            # it is crucial to use paddings in case of tracks with different lengths
            n_stations = max(pred_track.shape[0], target_track.shape[0])
            target_track_padded = np.zeros(
                (n_stations, target_track.shape[1]), dtype=np.float32)
            target_track_padded[:len(target_track)] = target_track
            pred_track_padded = np.zeros(
                (n_stations, pred_track.shape[1]), dtype=np.float32)
            pred_track_padded[:len(pred_track)] = pred_track
            # convert to tensors
            pred_track_padded = torch.from_numpy(pred_track_padded)
            target_track_padded = torch.from_numpy(target_track_padded)
            # calculate dists
            # TODO: calculate mean instead of max???
            dists += self._dist(pred_track_padded, target_track_padded).max()

        return {"loss_hits": dists / (len(predicted_tracks) + 1)}

    def _dist(self, x_1, x_2):
        return torch.sqrt(
            (x_2[:, 0] - x_1[:, 0]) ** 2
            + (x_2[:, 1] - x_1[:, 1]) ** 2
            + (x_2[:, 2] - x_1[:, 2]) ** 2
        )

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(source, i)
             for i, (source, _) in enumerate(indices)]
        )
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(target, i)
             for i, (_, target) in enumerate(indices)]
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
                "params": targets[idx][..., :-1],
                "class_labels": targets[idx][..., -1],  # charge
            }
            for idx in range(targets.shape[0])
        ]
        # match param vectors
        # TODO: why without charge???
        indices = self.matcher(outputs_without_aux, targets[..., :-1])

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_tracks = targets.shape[0] * targets.shape[1]
        # sum(len(t["class_labels"]) for t in targets)
        num_tracks = torch.as_tensor(
            [num_tracks], dtype=torch.float, device=next(
                iter(outputs.values())).device
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
