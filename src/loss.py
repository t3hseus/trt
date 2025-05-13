from typing import Callable, Dict, Tuple, Union

import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import v_measure_score
from torch import Tensor, nn
from torch.nn import functional as F


def dice_loss(inputs: Tensor, targets: Tensor, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice


def focal_loss(inputs, targets, alpha=0.8, gamma=2):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    # first compute binary cross-entropy
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
    return focal_loss


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

def match_hit_masks(outputs: torch.Tensor, targets: torch.Tensor):
    preds = torch.sigmoid(outputs)
    cost_matrix = torch.cdist(preds.float(), targets.float(), p=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return row_ind, col_ind


def compute_hungarian_loss(
    outputs: Tensor, targets: Tensor, distance: Callable
) -> Tensor:
    # F.l1_loss  F.smooth_l1_loss  F.mse_loss
    return distance(outputs, targets)


def params_distance(outputs: Tensor, targets: Tensor) -> Tensor:
    return F.l1_loss(outputs, targets)


def compute_mask_loss(
    outputs: Tensor, targets: Tensor, dice_coeff: float = 10., focal_coeff: float = 20.
) -> Tensor:
    inputs = F.sigmoid(outputs)
    return dice_coeff * dice_loss(inputs, targets) + focal_coeff * focal_loss(inputs, targets)


def compute_vertex_distance(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: tuple[float] = (0.1, 0.1, 0.8),
) -> torch.Tensor:
    outputs = outputs.squeeze()
    vertex_target = targets[:, 0, :3].squeeze()
    weights_ = torch.tensor(weights, device=outputs.device, requires_grad=False)
    return torch.nn.functional.l1_loss(outputs * weights_, vertex_target * weights_)


class TRTHungarianLoss(nn.Module):
    def __init__(
        self,
        params_distance: Callable = params_distance,
        class_loss: Callable = F.cross_entropy,
        segmentation_loss: Callable = F.binary_cross_entropy_with_logits,
        weights: tuple[float, ...] = (1, 1, 1, 1, 1),
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
        self.last_batch_match: Union[list[tuple[int, int]],  None] = []

    def _calc_loss(
        self,
        pred_params: Tensor,
        pred_masks: Tensor,
        target_params: Tensor,
        target_masks: Tensor,
        preds_lengths: Tensor,
        targets_lengths: Tensor,
        pred_logits: Tensor,
        target_labels: Tensor,
        batch_size: int,
        preds_segmentation_logits: Tensor,
        target_segmentation_labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        hungarian_loss = torch.tensor(0.0).to(pred_params.device)
        label_loss = torch.tensor(0.0).to(pred_params.device)
        segmentation_loss = torch.tensor(0.0).to(pred_params.device)
        mask_loss = torch.tensor(0.0).to(pred_params.device)

        if not self.params_with_vertex:
            target_params = target_params[..., 3:]
        transposed_masks = pred_masks.transpose(-1, -2)
        #self.save_match(batch_size,pred_params,target_params,preds_lengths,targets_lengths)
        self.save_match(batch_size, transposed_masks, target_masks, preds_lengths, targets_lengths)
        for i in range(batch_size):
            row_ind, col_ind = self.last_batch_match[i]
            matched_outputs = pred_params[i, row_ind]
            matched_targets = target_params[i, col_ind]
            matched_masks = transposed_masks[i, row_ind]
            matched_target_masks = target_masks[i, col_ind]

            hungarian_loss += compute_hungarian_loss(
                matched_outputs, matched_targets, distance=self._params_distance
            )
            mask_loss += compute_mask_loss(matched_masks, matched_target_masks)

            matched_targets = adjust_targets(
                row_ind=row_ind,
                col_ind=col_ind,
                targets=target_labels[i, : targets_lengths[i]],
                num_candidates=pred_logits.shape[1],
            )
            label_loss += self._class_loss_func(pred_logits[i], matched_targets)
            segmentation_loss += self._segmentation_loss_func(
                preds_segmentation_logits[i].squeeze(-1), target_segmentation_labels[i]
            )

        return hungarian_loss, mask_loss, label_loss, segmentation_loss

    def save_match(
            self,
            batch_size,
            preds,
            targets,
            preds_lengths,
            targets_lengths,

    ):
        for i in range(batch_size):
            #row_ind, col_ind = match_targets(
            #    outputs=pred_params[i, : preds_lengths[i]],
            #    targets=target_params[i, : targets_lengths[i]],
            #)
            row_ind, col_ind = match_hit_masks(
                outputs=preds[i, : preds_lengths[i]],
                targets=targets[i, : targets_lengths[i]].float(),
            )

            self.last_batch_match.append((row_ind, col_ind))

    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
        preds_lengths: Tensor,
        targets_lengths: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        self.last_batch_match = []
        batch_size = preds["params"].shape[0]
        pred_logits = preds["logits"]
        target_labels = targets["labels"]
        pred_params = preds["params"]
        target_params = targets["targets"]
        pred_masks = preds["track_hits_logits"]
        target_masks = targets["hit_track_masks"]
        preds_segmentation_logits = preds["fake_hit_logits"]
        target_segmentation_labels = (targets["hit_labels"] > -1).to(torch.float)
        if not self.intermediate:
            hungarian_loss, mask_loss, label_loss, segmentation_loss = self._calc_loss(
                pred_params=pred_params,
                target_params=target_params,
                pred_masks=pred_masks,
                target_masks=target_masks,
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
            mask_loss = torch.tensor(0.0).to(pred_params.device)
            for step in range(pred_params.shape[0]):
                hungarian_loss_step, mask_loss_step, label_loss_step, segmentation_loss_step = (
                    self._calc_loss(
                        pred_params=pred_params[step],
                        target_params=target_params,
                        pred_masks=pred_masks,
                        target_masks=pred_masks,
                        preds_lengths=preds_lengths,
                        targets_lengths=targets_lengths,
                        pred_logits=pred_logits[step],
                        target_labels=target_labels,
                        preds_segmentation_logits=preds_segmentation_logits,
                        target_segmentation_labels=target_segmentation_labels,
                        batch_size=batch_size,
                    )
                )
                mask_loss += mask_loss_step
                hungarian_loss += hungarian_loss_step
                label_loss += label_loss_step
                segmentation_loss += segmentation_loss_step

        hungarian_loss /= batch_size
        mask_loss /= batch_size
        label_loss /= batch_size
        segmentation_loss /= batch_size

        vertex_loss = compute_vertex_distance(
            preds["vertex"].unsqueeze(1), targets["targets"]
        )

        total_loss = (
            self._weights[0] * hungarian_loss
            + self._weights[1] * label_loss
            + self._weights[2] * vertex_loss
            + self._weights[3] * segmentation_loss
            + self._weights[4] * mask_loss
        )
        loss_components = {
            "params_dist": hungarian_loss.cpu().detach().item(),
            "matching_loss": label_loss.cpu().detach().item(),
            "vertex_dist": vertex_loss.cpu().detach().item(),
            "segmentation_loss": segmentation_loss.cpu().detach().item(),
            "mask_loss": mask_loss.cpu().detach().item(),
        }

        return total_loss, loss_components



class BaselineLoss(nn.Module):
    def __init__(
        self, segmentation_loss: Callable = F.binary_cross_entropy_with_logits
    ) -> None:
        super().__init__()

        self._segmentation_loss_func = segmentation_loss

    def forward(
        self,
        preds: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        **_,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_size = preds["hit_logits"].shape[0]
        device = preds["hit_logits"].device
        preds_segmentation_logits = preds["hit_logits"].squeeze(-1)
        target_segmentation_labels = (targets["hit_labels"] > -1).to(torch.float)
        target_cluster_labels = targets["hit_labels"].detach().cpu().numpy()
        segmentation_loss = torch.tensor(0.0).to(device)
        clustering_score = torch.tensor(0.0).to(device)
        for i in range(batch_size):
            segmentation_loss += self._segmentation_loss_func(
                preds_segmentation_logits[i], target_segmentation_labels[i]
            )

            if "cluster_labels" in preds:
                mask = (
                    preds["hit_logits"][i].sigmoid() > 0.5
                ).detach().cpu().numpy().squeeze(-1)
                true_labels = target_cluster_labels[i, mask]
                clustering_score += v_measure_score(
                    labels_true=true_labels, labels_pred=preds["cluster_labels"][i]
                )

        segmentation_loss /= batch_size
        clustering_score /= batch_size

        additional_losses = {}
        if "cluster_labels" in preds:
            additional_losses["clustering_loss"] = clustering_score

        return segmentation_loss, additional_losses


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
