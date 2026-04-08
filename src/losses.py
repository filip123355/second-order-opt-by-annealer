import torch
import torch.nn as nn

from .models import SVM, Ridge


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
        
        assert isinstance(model, Ridge), "Model must be an instance of Ridge for RidgeLoss."

        if targets.shape != outputs.shape:
            targets = targets.view_as(outputs)

        mse_loss = nn.MSELoss(reduction="mean")(outputs, targets)
        ridge_penalty = self.alpha * torch.sum(model.linear.weight ** 2)
        return mse_loss + ridge_penalty
    

class SVMSquaredHingeLoss(nn.Module):

    def __init__(self, 
                 C: float = 1.0,
    ) -> None:
        super(SVMSquaredHingeLoss, self).__init__()
        self.C = C

    def forward(self, 
                outputs: torch.Tensor,
                y: torch.Tensor,
                model: nn.Module,
    ) -> torch.Tensor:
        
        assert isinstance(model, SVM), "Model must be an instance of SVM for SVMSquaredHingeLoss." 

        if y.shape != outputs.shape:
            y = y.view_as(outputs)

        out = (torch.ones_like(y).to(y.device) - y * outputs) ** 2
        out = nn.ReLU()(out)
        return 1 / 2 * torch.sum(model.linear.weight ** 2) + self.C * 1 / 2 * torch.sum(out) 

        