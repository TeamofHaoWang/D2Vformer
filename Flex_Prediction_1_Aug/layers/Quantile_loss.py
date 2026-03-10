import torch
import torch.nn as nn

class QuantileLoss(nn.Module):

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # target.shape:bs,prelen,dim
        # preds.shape bs,prelen,dim,quantiles_number
        target = target.flatten()
        preds = preds.flatten().unsqueeze(-1).repeat(1,len(self.quantiles))
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss