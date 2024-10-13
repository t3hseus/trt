import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import L1Loss


def cardinality_error(
        pred_tracks, pred_logits, target_tracks, threshold: float=0.5
) -> int:
    pred_tracks = pred_tracks[pred_logits > threshold]
    return len(pred_tracks) - len(target_tracks)

def hits_distance(
        pred_tracks, pred_logits, target_tracks, threshold: float=0.5
) -> int:
    pred_tracks = pred_tracks[pred_logits > threshold]
    return len(pred_tracks) - len(target_tracks)


def match_targets(outputs, targets):
    cost_matrix = torch.cdist(outputs, targets, p=1)
    row_ind, col_ind = linear_sum_assignment(
        cost_matrix.cpu().detach().numpy()
    )
    return row_ind, col_ind


def vertex_distance(outputs, targets):
    #vertex_out = outputs[:, 0, :3]
    vertex_target = targets[:, 0, :3]
    return torch.nn.functional.l1_loss(outputs.squeeze(), vertex_target) * 3

