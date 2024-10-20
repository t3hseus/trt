from typing import Callable, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F


def adjust_targets(row_ind, col_ind, targets, num_candidates=10):
    """
    Args:
        # logits: Predicted logits with shape [num_candidates, num_classes]
        row_ind: Matched row indices for predictions N (for N matched pairs).
        col_ind: Matched column indices for predictions N.
        targets: Ground truth labels corresponding to matched pairs N.
        num_candidates (int): Number of candidates predicted per sample (default=10).

    Returns:
        # adjusted_logits: Logits with shape [num_candidates, num_classes],
            adjusted for unmatched candidates.
        adjusted_targets: Target labels with shape num_candidates,
            where unmatched candidates get label 1.
    """
    # Initialize adjusted logits and targets
    # adjusted_logits = logits  #.clone()  # Copy logits
    adjusted_targets = torch.ones(
        num_candidates, dtype=torch.long, device=targets.device
    )
    # Default label is 1 for unmatched candidates

    # For each matched pair, assign the corresponding target
    matched_rows = row_ind
    matched_cols = col_ind
    adjusted_targets[matched_rows] = targets[matched_cols]

    return adjusted_targets


def match_targets(outputs, targets):
    cost_matrix = torch.cdist(outputs, targets, p=1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return row_ind, col_ind


def compute_hungarian_loss(
    outputs: Tensor, targets: Tensor, distance: Callable
) -> Tensor:
    # F.l1_loss  F.smooth_l1_loss  F.mse_loss
    return distance(outputs, targets)


def params_distance(outputs: Tensor, targets: Tensor) -> Tensor:
    return F.l1_loss(outputs, targets)


def compute_vertex_distance(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: tuple[float] = (0.1, 0.1, 0.8),
) -> torch.Tensor:
    outputs = outputs.squeeze()
    vertex_target = targets[:, 0, :3]
    weights_ = torch.tensor(weights, device=outputs.device, requires_grad=False)
    return torch.nn.functional.l1_loss(outputs * weights_, vertex_target * weights_)


class TRTHungarianLoss(nn.Module):
    def __init__(
        self,
        params_distance: Callable = params_distance,
        class_loss: Callable = F.cross_entropy,
        segmentation_loss: Callable = F.cross_entropy,
        weights: tuple[float, ...] = (1, 1, 1, 1),
        intermediate: bool = False,
        params_with_vertex: bool = False,
    ):
        super().__init__()

        self.intermediate = intermediate
        self._params_distance = params_distance
        self._class_loss_func = class_loss
        self._segmentation_loss_func = segmentation_loss
        self._weights = weights
        self.params_with_vertex = params_with_vertex

    def _calc_loss(
        self,
        pred_params: Tensor,
        target_params: Tensor,
        preds_lengths: Tensor,
        targets_lengths: Tensor,
        pred_logits: Tensor,
        target_labels: Tensor,
        batch_size: int,
        preds_segmentation_logits: Optional[Tensor] = None,
        target_segmentation_labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        hungarian_loss = torch.tensor(0.0).to(pred_params.device)
        label_loss = torch.tensor(0.0).to(pred_params.device)
        segmentation_loss = torch.tensor(0.0).to(pred_params.device)

        if not self.params_with_vertex:
            target_params = target_params[..., 3:]

        for i in range(batch_size):
            row_ind, col_ind = match_targets(
                outputs=pred_params[i, : preds_lengths[i]],
                targets=target_params[i, : targets_lengths[i]],
            )
            matched_outputs = pred_params[i, row_ind]
            matched_targets = target_params[i, col_ind]
            hungarian_loss += compute_hungarian_loss(
                matched_outputs, matched_targets, distance=self._params_distance
            )

            matched_targets = adjust_targets(
                row_ind=row_ind,
                col_ind=col_ind,
                targets=target_labels[i, : targets_lengths[i]],
                num_candidates=pred_logits.shape[1],
            )
            label_loss += self._class_loss_func(pred_logits[i], matched_targets)

            if (
                preds_segmentation_logits is not None
                and target_segmentation_labels is not None
            ):
                segmentation_loss += self._segmentation_loss_func(
                    preds_segmentation_logits[i], target_segmentation_labels[i]
                )

        return hungarian_loss, label_loss, segmentation_loss

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
        preds_lengths: Tensor,
        targets_lengths: Tensor,
    ) -> Tensor:
        batch_size = preds["params"].shape[0]
        pred_logits = preds["logits"]
        target_labels = targets["labels"]
        pred_params = preds["params"]
        target_params = targets["targets"]
        preds_segmentation_logits = (
            preds["hit_logits"] if "hit_logits" in preds else None
        )
        target_segmentation_labels = (
            targets["hit_labels"] if "hit_labels" in targets else None
        )
        if not self.intermediate:
            hungarian_loss, label_loss, segmentation_loss = self._calc_loss(
                pred_params=pred_params,
                target_params=target_params,
                preds_lengths=preds_lengths,
                targets_lengths=targets_lengths,
                pred_logits=pred_logits,
                target_labels=target_labels,
                preds_segmentation_logits=preds_segmentation_logits,
                target_segmentation_labels=target_segmentation_labels,
                batch_size=batch_size,
            )
        else:
            # TODO
            hungarian_loss = torch.tensor(0.0).to(pred_params.device)
            label_loss = torch.tensor(0.0).to(pred_params.device)
            segmentation_loss = torch.tensor(0.0).to(pred_params.device)

            for step in range(pred_params.shape[0]):
                hungarian_loss_step, label_loss_step, segmentation_loss_step = (
                    self._calc_loss(
                        pred_params=pred_params[step],
                        target_params=target_params,
                        preds_lengths=preds_lengths,
                        targets_lengths=targets_lengths,
                        pred_logits=pred_logits[step],
                        target_labels=target_labels,
                        preds_segmentation_logits=preds_segmentation_logits,
                        target_segmentation_labels=target_segmentation_labels,
                        batch_size=batch_size,
                    )
                )
                hungarian_loss += hungarian_loss_step
                label_loss += label_loss_step
                segmentation_loss += segmentation_loss_step

        hungarian_loss /= batch_size
        label_loss /= batch_size
        segmentation_loss /= batch_size

        vertex_loss = compute_vertex_distance(
            preds["vertex"].unsqueeze(1), targets["targets"]
        )

        return (
            self._weights[0] * hungarian_loss
            + self._weights[1] * label_loss
            + self._weights[2] * vertex_loss
            + self._weights[3] * segmentation_loss
        )


if __name__ == "__main__":
    trt_loss = TRTHungarianLoss()
    preds_coord = torch.rand((16, 25, 4))
    preds_vertex = torch.rand((16, 3))
    preds_labels = torch.rand((16, 25, 2))

    targets_dict = torch.rand((16, 10, 7))
    preds_coord[:, :10] = targets_dict[:, :, 3:]

    target_vertex = torch.rand((8, 3)).unsqueeze(1).repeat(1, 10, 1)
    target_vertex_1 = preds_vertex[8:, :].unsqueeze(1).repeat(1, 10, 1)
    targets_dict[:8, :, :3] = target_vertex
    targets_dict[8:, :, :3] = target_vertex_1

    target_labels = torch.ones((16, 10), dtype=torch.long)
    target_lengths = [10 for i in range(16)]
    pred_lengths = [25 for i in range(16)]

    preds_dict = {"params": preds_coord, "vertex": preds_vertex, "logits": preds_labels}
    targets_dict = {"targets": targets_dict, "labels": target_labels}
    print(
        trt_loss(
            preds_dict,
            targets_dict,
            preds_lengths=pred_lengths,
            targets_lengths=target_lengths,
        )
    )
