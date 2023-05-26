import torch
from torch import nn

torch_bce = nn.BCELoss(reduction="none")


# bce
def bce(x, y):
    return torch_bce(x[0], y)


def sup_bce(x, y):
    s = 0
    for p in x[1:]:
        s += torch_bce(p, y)
    return s


# jaccard
def soft_jaccard(x, y):
    eps = 0.0000001
    return ((x * y).sum([-1, -2]) + eps) / (x.sum([-1, -2]) + y.sum([-1, -2]) - (x * y).sum([-1, -2]) + eps)


def jaccard_loss(x, y):
    return 1 - soft_jaccard(x[0], y)


def sup_jaccard_loss(x, y):
    s = 0
    for p in x[1:]:
        s += 1 - soft_jaccard(p, y)
    return s


def inv_soft_jaccard(x, y):
    return soft_jaccard(1 - x[0], 1 - y)


def inv_jaccard_loss(x, y):
    return 1 - inv_soft_jaccard(x[0], y)


def sup_inv_jaccard_loss(x, y):
    s = 0
    for p in x[1:]:
        s += 1 - soft_jaccard(p, y)
    return s


# dice
def soft_dice(x, y):
    eps = 0.0000001
    return 2 * (((x * y).sum([-1, -2]) + eps) / (x.sum([-1, -2]) + y.sum([-1, -2])) + eps)


def ln_dice(x, y):
    return -torch.log(soft_dice(x[0], y))


def sup_ln_dice(x, y):
    s = 0
    for p in x[1:]:
        s += -torch.log(soft_dice(p, y))
    return s


def inv_soft_dice(x, y):
    return soft_dice(1 - x, 1 - y)


def inv_ln_dice(x, y):
    return -torch.log(inv_soft_dice(x[0], y))


def sup_inv_ln_dice(x, y):
    s = 0
    for p in x[1:]:
        s += -torch.log(inv_soft_dice(p, y))
    return s


name2func = {
    # Loss
    "bce": bce,
    "sup_bce": sup_bce,

    "jaccard_loss": jaccard_loss,
    "inv_jaccard_loss": inv_jaccard_loss,
    "sup_jaccard_loss": sup_jaccard_loss,

    "ln_dice": ln_dice,
    "sup_ln_dice": sup_ln_dice,

    "inv_ln_dice": inv_ln_dice,
    "sup_inv_ln_dice": sup_inv_ln_dice,

    # Metrics
    "soft_jaccard": soft_jaccard,
    "inv_soft_jaccard": inv_soft_jaccard,

    "soft_dice": soft_dice,
    "inv_soft_dice": inv_soft_dice,

}
