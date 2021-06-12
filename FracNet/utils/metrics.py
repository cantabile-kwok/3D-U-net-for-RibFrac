def dice(x, y, image=False):
    x = x.sigmoid()
    i, u = [t.flatten(1).sum(1) if image else t.sum() for t in [x * y, x + y]]
    dc = ((2 * i + 1) / (u + 1)).mean()
    return dc


def recall(x, y, thresh=0.1):
    x = x.sigmoid()
    tp = (((x * y) > thresh).flatten(1).sum(1) > 0).sum()
    rc = tp / (((y > 0).flatten(1).sum(1) > 0).sum() + 1e-8)
    return rc


def accuracy(x, y, thresh=0.5):
    x = x.sigmoid()
    ac = ((x > thresh) == (y > 0)).float().mean()
    return ac


def precision(x, y, thresh=0.1):
    x = x.sigmoid()
    tp = (((x * y) > thresh).flatten(1).sum(1) > 0).sum()
    pc = tp / (((x > thresh).flatten(1).sum(1) > 0).sum() + 1e-8)
    return pc


def fbeta_score(x, y, beta=1, **kwargs):
    rc = recall(x, y, **kwargs)
    pc = precision(x, y, **kwargs)
    fs = (1 + beta ** 2) * pc * rc / (beta ** 2 * pc + rc + 1e-8)
    return fs
