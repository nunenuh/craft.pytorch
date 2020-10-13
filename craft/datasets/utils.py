import numpy as np
import cv2 as cv
from typing import *
import json



def load_image(impath):
    img = cv.imread(impath)
    if type(img) is type(None):  # always check for None
        raise ValueError(f"unable to load Image {str(impath)}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img


def load_json2dict(pathfile):
    with open(pathfile) as jsfile:
        data = json.load(jsfile)
    return data

def combine_point_single(bx: np.ndarray, by: np.ndarray, idxpos: int, to_int: bool = False):
    if to_int:
        return int(bx[idxpos]), int(by[idxpos])
    else:
        return bx[idxpos], by[idxpos]


def combine_point_multi(bx: np.ndarray, by: np.ndarray, idx: int, idxpos: int, to_int: bool = False):
    if to_int:
        return int(bx[idx][idxpos]), int(by[idx][idxpos])
    else:
        return bx[idx][idxpos], by[idx][idxpos]


def raw2box(bx: np.ndarray, by: np.ndarray, to_int: bool = False):
    tl, tr, br, bl = 0, 1, 2, 3
    btl = combine_point_single(bx, by, tl, to_int)
    btr = combine_point_single(bx, by, tr, to_int)
    bbr = combine_point_single(bx, by, br, to_int)
    bbl = combine_point_single(bx, by, bl, to_int)
    return [btl, btr, bbr, bbl]


def raw2boxmulti(bx: np.ndarray, by: np.ndarray, idx: int, to_int: bool = False):
    tl, tr, br, bl = 0, 1, 2, 3
    btl = combine_point_multi(bx, by, idx, tl, to_int)
    btr = combine_point_multi(bx, by, idx, tr, to_int)
    bbr = combine_point_multi(bx, by, idx, br, to_int)
    bbl = combine_point_multi(bx, by, idx, bl, to_int)
    return [btl, btr, bbr, bbl]


def raw2bbox(data: np.ndarray, to_int: bool = False):
    boxes = []
    x, y, tl, tr, br, bl = 0, 1, 0, 1, 2, 3
    bx, by = data[x].T, data[y].T
    if len(bx.shape) == 2:
        for idx in range(len(bx)):
            box = raw2boxmulti(bx, by, idx, to_int)
            boxes.append(box)
    else:
        box = raw2box(bx, by, to_int)
        boxes.append(box)

    return boxes
