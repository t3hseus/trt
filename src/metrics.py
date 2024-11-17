import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import L1Loss


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


def hits_dist(preds, targets):
    "Dist based on DTW (we get time-based sequences of hits)"
    n, m = len(preds), len(targets)

    dtw_matrix = torch.zeros((n + 1, m + 1), dtype=torch.float32)
    dtw_matrix[0, 1:] = float("inf")
    dtw_matrix[1:, 0] = float("inf")
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (preds[i - 1] - targets[j - 1]) ** 2
            last_min = torch.min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]
