import torch
import torch.nn as nn


class RidgeLoss(nn.Module):
    def __init__(self, 
                 alpha: float = 1.0,
    ) -> None:
        super(RidgeLoss, self).__init__()   
        self.alpha = alpha
    def forward(self, 
                outputs: torch.Tensor, 
                targets: torch.Tensor,
                model: nn.Module,
    ) -> torch.Tensor:
        if targets.shape != outputs.shape:
            targets = targets.view_as(outputs)

        mse_loss = nn.MSELoss(reduction="mean")(outputs, targets)
        ridge_penalty = self.alpha * torch.sum(model.linear.weight ** 2)
        return mse_loss + ridge_penalty