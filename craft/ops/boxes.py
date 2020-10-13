from typing import *

import numpy as np
from shapely.geometry import Polygon


def coord2xymm(box, to_int=False):
    tl, tr, br, bl = box
    x_idx, y_idx = 0, 1
    xmin, xmax, ymin, ymax,  = tl[x_idx], br[x_idx], tl[y_idx], bl[y_idx]
    if to_int:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    return xmin, ymin, xmax, ymax


def xymm2coord(box, to_int=False):
    xmin, ymin, xmax, ymax = box
    if to_int:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    four_points = [
        [xmin, ymin],  # top left
        [xmax, ymin],  # top right
        [xmax, ymax],  # bottom right
        [xmin, ymax],  # bottom left
    ]
    return four_points


def coord2xywh(box):
    xmin, ymin, xmax, ymax = coord2xymm(box)
    return xmin, ymin, xmax - xmin, ymax - ymin


def batch_coord2xymm(boxes, to_int=False):
    xymm_boxes = []
    for box in boxes:
        res = coord2xymm(box, to_int=to_int)
        res = np.array(res).astype(np.float32)
        xymm_boxes.append(res)

    return np.array(xymm_boxes).astype(np.float32)


def batch_xymm2coord(boxes, to_int=False):
    four_points = []
    for box in boxes:
        res = xymm2coord(box, to_int=to_int)
        res = np.array(res).astype(np.float32)
        four_points.append(res)
    return four_points


def pad(box, factor=0.1, to_int=True):
    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin
    wf, hf = w * factor, h * factor
    xmin, ymin, xmax, ymax = xmin - wf, ymin - hf, xmax + wf, ymax + hf
    box_out = [xmin, ymin, xmax, ymax]
    if to_int:
        box_out = [int(xmin), int(ymin), int(xmax), int(ymax)]

    # check if minus set to zero
    for idx, val in enumerate(box_out):
        if val < 0:
            box_out[idx] = 0

    return tuple(box_out)


def centroid(box: List[tuple]):
    pbox = Polygon(box).centroid
    return int(pbox.x), int(pbox.y)


def order(box: List[tuple], use_pad=False, pad_factor=0.1):
    xmin, ymin, xmax, ymax = bounds(
        box, use_pad=use_pad, pad_factor=pad_factor)
    tl, tr, br, bl = (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
    return [tl, tr, br, bl]


def bounds(box: List[tuple], use_pad=False, pad_factor=0.1):
    bounds = Polygon(box).bounds
    xmin, ymin, xmax, ymax = bounds
    if use_pad:
        xmin, ymin, xmax, ymax = pad(bounds, factor=pad_factor)
    return [int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))]


def delta_xy(box: List[tuple], use_pad=False, pad_factor=0.1):
    xmin, ymin, xmax, ymax = bounds(
        box, use_pad=use_pad, pad_factor=pad_factor)
    dx = xmax - xmin
    dy = ymax - ymin
    return dx, dy


def triangle_centroid(box_center, bp_left, bp_right):
    tcenter = Polygon([box_center, bp_left, bp_right]).centroid
    return int(tcenter.x), int(tcenter.y)


def center_triangle_top_bottom(box: List[tuple]):
    box_tl, box_tr, box_br, box_bl = box
    box_center = centroid(box)
    tct = triangle_centroid(box_center, box_tl, box_tr)
    tcb = triangle_centroid(box_center, box_bl, box_br)
    return tct, tcb


def iou(boxa: List, boxb: List):
    a = Polygon(boxa)
    b = Polygon(boxb)
    intersect_area = a.intersection(b).area
    union_area = a.union(b).area
    return intersect_area / union_area
