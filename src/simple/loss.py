from typing import Callable

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from src.metrics import vertex_distance


def hungarian_loss_with_match(outputs, targets):
    row_ind, col_ind = match_targets(outputs, targets)
    matched_outputs = outputs[row_ind]
    matched_targets = targets[col_ind]
    loss_dist = F.l1_loss(matched_outputs, matched_targets)
    return loss_dist


def criterion(preds, targets, preds_lengths, targets_lengths):
    hungarian = torch.tensor(0.0)
    for i in range(preds.shape[0]):
        hungarian += hungarian_loss_with_match(
            preds[i, :preds_lengths[i]],
            targets[i, :targets_lengths[i]]
        )
    hungarian /= preds.shape[0]  # batchmean
    return hungarian


def adjust_targets(row_ind, col_ind, targets, num_candidates=10):
    """
    Args:
        logits: Predicted logits with shape [num_candidates, num_classes]
        row_ind: Matched row indices for predictions N (for N matched pairs).
        col_ind: Matched column indices for predictions N.
        targets: Ground truth labels corresponding to matched pairs N.
        num_candidates (int): Number of candidates predicted per sample (default=10).

    Returns:
        adjusted_logits: Logits with shape [num_candidates, num_classes], adjusted for unmatched candidates.
        adjusted_targets: Target labels with shape num_candidates, where unmatched candidates get label 1.
    """
    # Initialize adjusted logits and targets
    #adjusted_logits = logits  #.clone()  # Copy logits
    adjusted_targets = torch.ones(num_candidates, dtype=torch.long, device=targets.device)
    # Default label is 1 for unmatched candidates

    # For each matched pair, assign the corresponding target
    matched_rows = row_ind
    matched_cols = col_ind
    adjusted_targets[matched_rows] = targets[matched_cols]

    return adjusted_targets


def focal_loss(logits, targets, alpha=1, gamma=2, reduction='mean'):
    """
    Args:
        logits: Predictions for each class with shape [B, N, C] where C is the number of classes (raw logits, not softmaxed).
        targets: Ground truth labels with shape [B, N] where each value is in the range [0, C-1].
        alpha (float, optional): A balancing factor for classes (default=1).
        gamma (float, optional): Focusing parameter for hard examples (default=2).
        reduction (string, optional): Specifies the reduction to apply to the output:
                                      'none' | 'mean' | 'sum'. 'mean': the sum of the output will be divided by the number of elements in the output;
                                      'sum': the output will be summed;
                                      'none': no reduction will be applied (default='mean').
    Returns:
        Loss: Scalar if reduction is applied or the same shape as input without reduction.
    """
    probs = F.softmax(logits, dim=-1)  # [N, C]

    targets = targets.unsqueeze(-1)  # [N, 1] to align with logits
    probs_target_class = probs.gather(dim=-1, index=targets).squeeze(-1)  # [N]

    # Compute the focal loss
    log_pt = torch.log(probs_target_class + 1e-9)  # Stability for log
    loss = -alpha * (1 - probs_target_class) ** gamma * log_pt  # Focal loss equation

    # Apply the reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # No reduction


def match_targets(outputs, targets):
    cost_matrix = torch.cdist(outputs, targets, p=1)
    row_ind, col_ind = linear_sum_assignment(
        cost_matrix.cpu().detach().numpy()
    )
    return row_ind, col_ind


def hungarian_loss(outputs, targets, distance: Callable, num_params: int = 7):

    # loss = F.l1_loss(matched_outputs, matched_targets)
    # loss = F.smooth_l1_loss(matched_outputs, matched_targets)
    # loss = F.mse_loss(matched_outputs, matched_targets)
    loss = distance(outputs, targets)  # averaging over all elements
    return loss * num_params  # to maintain distance logic


class TRTHungarianLoss(nn.Module):
    def __init__(
            self,
            distance: Callable = F.l1_loss,
            class_loss: Callable = F.cross_entropy,
            weights: tuple[float, float, float] = (1, 1, 1),
            intermediate: bool = False
    ):
        super().__init__()
        self.intermediate = intermediate
        self._distance = distance
        self._class_loss = class_loss
        self._weights = weights

    def _calc_loss(
            self,
            pred_params, target_params,
            preds_lengths, targets_lengths,
            pred_logits, target_labels,
            batch_size
    ):
        hungarian = torch.tensor(0.0).to(pred_params.device)
        label_loss = torch.tensor(0.0).to(pred_params.device)
        for i in range(batch_size):
            row_ind, col_ind = match_targets(
                pred_params[i, :preds_lengths[i]],
                target_params[i, :targets_lengths[i]])
            matched_outputs = pred_params[i, row_ind]
            matched_targets = target_params[i, col_ind]
            hungarian += hungarian_loss(
                matched_outputs,
                matched_targets,
                distance=self._distance
            )
            matched_targets = adjust_targets(
                row_ind,
                col_ind,
                target_labels[i, :targets_lengths[i]],
                num_candidates=pred_logits.shape[1]
            )
            label_loss += class_loss(
                pred_logits[i],
                matched_targets,
                loss_fn=self._class_loss
            )
        return hungarian, label_loss

    def forward(
            self,
            preds: dict[str, torch.Tensor],
            targets: dict[str, torch.Tensor],
            preds_lengths,
            targets_lengths,

    ):
        batch_size = preds["params"].shape[0]
        pred_logits = preds["logits"]
        pred_params = preds["params"]
        target_params = targets["targets"]
        target_labels = targets["labels"]
        if not self.intermediate:
            hungarian, label_loss = self._calc_loss(
                pred_params, target_params,
                preds_lengths, targets_lengths,
                pred_logits, target_labels,
                batch_size
            )
        else:
            hungarian = torch.tensor(0.).to(pred_params.device)
            label_loss = torch.tensor(0.).to(pred_params.device)
            for step in range(pred_params.shape[0]):
                hungarian_step, label_loss_step = self._calc_loss(
                    pred_params[step], target_params,
                    preds_lengths, targets_lengths,
                    pred_logits[step], target_labels,
                    batch_size
                )
                hungarian += hungarian_step
                label_loss += label_loss_step

        vertex_loss = vertex_distance(preds["vertex"].unsqueeze(1), targets["targets"])
        hungarian /= batch_size  # batchmean
        label_loss /= batch_size  # batchmean
        vertex_loss /= batch_size
        return self._weights[0] * hungarian + self._weights[1] * label_loss + self._weights[2] * vertex_loss


def class_loss(outputs, targets, loss_fn: Callable):
    return loss_fn(outputs, targets)
