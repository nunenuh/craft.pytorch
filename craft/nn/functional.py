import torch
from typing import *

from torch import Tensor


def ohem_number(score: torch.Tensor, thresh: float = 0.1, k_ratio: int = 3) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    find sum of positive pixel and sum of negative pixel from label or predicted region or affinity
    :param score: predicted region or predicted affinity or labeled region or labeled score
    :param thresh: threshold number to find the gaussian heatmap, proper value is 0.1
    :param k_ratio: positive - negative ratio, proper value is 1:3
    :return: (sum of positive pixel, sum of negative pixel)
    """
    pospix: Tensor = torch.gt(score, thresh)
    negpix: Tensor = torch.le(score, thresh)
    sum_pospix, sum_negpix = torch.sum(pospix), torch.sum(negpix)

    # balancing the number of sum negative pixel
    if sum_pospix * k_ratio < sum_negpix: sum_negpix = sum_pospix * k_ratio
    return pospix, negpix, sum_pospix, sum_negpix


def positive_mask(score: torch.Tensor, thresh: float = 0):
    return torch.gt(score, thresh)


def negative_mask(score: torch.Tensor, thresh: float = 0):
    return torch.le(score, thresh)


def positive_mask_loss(base_loss: torch.Tensor, score: torch.Tensor, thresh: float = 0):
    mask = torch.gt(score, thresh)
    num = torch.sum(mask)
    loss = torch.masked_select(base_loss, mask)
    return loss, num


def negative_mask_loss(base_loss: torch.Tensor, score: torch.Tensor, thresh: float = 0):
    mask = torch.le(score, thresh)
    num = torch.sum(mask)
    loss = torch.masked_select(base_loss, mask)
    return loss, num


if __name__ == '__main__':
    # ohem_number()
    pass
