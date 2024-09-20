import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


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
        out_param = outputs["params"].flatten(0, 1)  # [batch_size * num_queries, 7]

        # Also concat the target labels and params
        # target_ids = torch.cat([v["class_labels"] for v in targets])
        # dummy targets
        target_ids = torch.ones(targets.shape[0] * targets.shape[1], dtype=torch.int)
        target_param = targets.flatten(0, 1)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class]. 1 may be omitted
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between parameters
        params_cost = torch.cdist(
            out_param.to(torch.float32), target_param.to(torch.float32), p=1
        )

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
