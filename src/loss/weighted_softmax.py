import torch
import torch.nn.functional as F

class WeightedSoftmaxLoss:
    def __call__(self, logits, targets, weights):
        ce = F.cross_entropy(logits, targets, reduction="none")
        weighted = ce * weights
        return weighted.sum() / (weights.sum() + 1e-8)
