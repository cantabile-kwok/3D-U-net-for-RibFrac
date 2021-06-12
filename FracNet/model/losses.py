from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MixLoss', 'DiceLoss', 'GHMCLoss', 'FocalLoss']


class MixLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x, y):
        lf, lfw = [], []
        for i, v in enumerate(self.args):
            if i % 2 == 0:
                lf.append(v)
            else:
                lfw.append(v)
        mx = sum([w * l(x, y) for l, w in zip(lf, lfw)])
        return mx


class DiceLoss(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image

    def forward(self, x, y):
        x = x.sigmoid()
        i, u = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x * y, x + y]]

        dc = (2 * i + 1) / (u + 1)
        dc = 1 - dc.mean()
        return dc


class GHMCLoss(nn.Module):
    def __init__(self, mmt=0, bins=10):
        super().__init__()
        self.mmt = mmt
        self.bins = bins
        self.edges = [x / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6

        if mmt > 0:
            self.acc_sum = [0] * bins

    def forward(self, x, y):
        w = torch.zeros_like(x)
        g = torch.abs(x.detach().sigmoid() - y)

        n = 0
        t = reduce(lambda x, y: x * y, w.shape)
        for i in range(self.bins):
            ix = (g >= self.edges[i]) & (g < self.edges[i + 1]); nb = ix.sum()
            if nb > 0:
                if self.mmt > 0:
                    self.acc_sum[i] = self.mmt * self.acc_sum[i] + (1 - self.mmt) * nb
                    w[ix] = t / self.acc_sum[i]
                else:
                    w[ix] = t / nb
                n += 1
        if n > 0:
            w = w / n

        gc = F.binary_cross_entropy_with_logits(x, y, w, reduction='sum') / t
        return gc


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        ce = F.binary_cross_entropy_with_logits(x, y)
        fc = self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce
        return fc
