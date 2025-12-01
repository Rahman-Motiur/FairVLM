import torch
import torch.nn as nn
import torch.nn.functional as F


class FairnessCalibratedLoss(nn.Module):
    """
    FCL: Combines base segmentation loss + demographic disparity penalties
    + counterfactual prompt regularization.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, gt):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * gt).sum(dim=[1, 2, 3])
        union = pred.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3])
        return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()

    def disparity_penalty(self, pred, gt, demographic):
        """
        Measures fairness gap across demographic groups.
        """
        groups = demographic.argmax(dim=1)
        losses = []

        for g in torch.unique(groups):
            idx = groups == g
            if idx.sum() == 0:
                continue
            loss_g = self.dice_loss(pred[idx], gt[idx])
            losses.append(loss_g)

        if len(losses) <= 1:
            return 0.0

        losses = torch.stack(losses)
        return losses.max() - losses.min()

    def forward(self, pred, gt, demographic):
        base = self.bce(pred, gt) + self.dice_loss(pred, gt)
        fairness = self.disparity_penalty(pred, gt, demographic)
        return base + 0.5 * fairness
