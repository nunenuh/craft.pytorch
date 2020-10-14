
from typing import *

import numpy as np
from shapely.geometry import Polygon
from . import boxes as boxez


def _affine_box(fbox: List[tuple], sbox: List[tuple], use_pad: bool = False, pad_factor: float = 0.2):
    fbox_tct, fbox_tcb = boxez.center_triangle_top_bottom(fbox)
    sbox_tct, sbox_tcb = boxez.center_triangle_top_bottom(sbox)

    tl, tr = fbox_tct, sbox_tct
    br, bl = sbox_tcb, fbox_tcb
    afbox = [tl, tr, br, bl]
    afbox = boxez.order(afbox, use_pad=use_pad, pad_factor=pad_factor)
    return afbox


def _affine_boxes(boxez: List[List], use_pad: bool = False, pad_factor: float = 0.2):
    out = []
    c = len(boxez)
    for i in range(c):
        if (i + 1) < c:
            fbox, sbox = boxez[i], boxez[i + 1]
            aff_box = _affine_box(fbox, sbox, use_pad=use_pad, pad_factor=pad_factor)
            out.append(aff_box)
    return out


def _is_box_intersect(boxa: List, boxb: List, threshold: float = 0.1):
    iou_result = boxez.iou(boxa, boxb)
    if iou_result > threshold:
        # print(iou_result)
        return True
    return False


def _find_word_over_char(word_bbox: List, char_bbox: List, threshold: float = 0.05) -> List:
    word_bbox, char_bbox = sorted(word_bbox), sorted(char_bbox)
    wboxes = []
    for i in range(len(word_bbox)):
        word = []
        for j in range(len(char_bbox)):
            iou = _is_box_intersect(word_bbox[i], char_bbox[j], threshold=threshold)
            if iou:
                word.append(char_bbox[j])
        wboxes.append(word)
    return wboxes


def boxes(wbbox: List, cbbox: List, threshold: float = 0.05,
                   use_pad: bool = False, pad_factor: float = 0.2):
    if type(wbbox) is not list:
        wbbox = wbbox.tolist()
        wbbox = sorted(wbbox)
    if type(cbbox) is not list:
        cbbox = cbbox.tolist()
        cbbox = sorted(cbbox)

    wboxes = _find_word_over_char(wbbox, cbbox, threshold=threshold)
    afboxes = []
    for wb in wboxes:
        afb = _affine_boxes(wb, use_pad=use_pad, pad_factor=pad_factor)
        afboxes = afboxes + afb
    return afboxes


if __name__ == '__main__':
    pass