import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import L1Loss

from torchmetrics.classification import Accuracy, Precision, Recall
from torchmetrics.segmentation import DiceScore


def mask_metrics(
       outputs: torch.Tensor, targets: torch.Tensor, match
) -> dict[str, float]:
    metrics = {"dice": 0.}
    preds = torch.sigmoid(outputs) > 0.5
    cum_len = 0
    for i in range(len(outputs)):
        row_ind, col_ind = match[i]
        matched_preds = preds[i][:, row_ind].unsqueeze(0)
        matched_targets = targets[i][:, col_ind].unsqueeze(0)
        cum_len += len(matched_preds)
        dice = DiceScore(
            len(preds),
            include_background=False,
            average="weighted"
        )(matched_preds, matched_targets).item()
        metrics["dice"] += dice
    metrics["dice"] /= cum_len
    return metrics


def cardinality_error(
    pred_tracks, pred_logits, target_tracks, threshold: float = 0.5
) -> int:
    pred_tracks = pred_tracks[pred_logits > threshold]
    return len(pred_tracks) - len(target_tracks)


def hits_distance(
    pred_tracks, pred_logits, target_tracks, threshold: float = 0.5
) -> int:
    pred_tracks = pred_tracks[pred_logits > threshold]
    return len(pred_tracks) - len(target_tracks)


def match_targets(outputs, targets):
    cost_matrix = torch.cdist(outputs, targets, p=1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return row_ind, col_ind


def vertex_distance(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: tuple[float] = (0.1, 0.1, 0.8),
) -> torch.Tensor:
    outputs = outputs.squeeze()
    vertex_target = targets[:, 0, :3]

    weights_ = torch.tensor(weights, device=outputs.device, requires_grad=False)
    # return torch.nn.functional.l1_loss(outputs, vertex_target) * 3
    return torch.nn.functional.l1_loss(outputs * weights_, vertex_target * weights_) * 3
