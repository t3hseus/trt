from torch import nn


class ParamDiff(nn.Module):
    def __init__(self, param_pos: int = 0):
        super().__init__()
        self.param_pos = param_pos

    def forward(self, preds, targets):
        return preds[:, :, self.param_pos] - targets[:, :, self.param_pos]


class Accuracy(nn.Module):
    def __init__(self, threshold: float = 1e-6):
        super().__init__()
        self.threshold = threshold

    def forward(self, preds, targets):
        se = nn.functional.mse_loss(preds, targets, reduce=False)
        return (se < self.threshold).sum() / len(se)
