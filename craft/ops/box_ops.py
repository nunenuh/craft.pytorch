from typing import *

import numpy as np
from shapely.geometry import Polygon


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def box_coordinate_to_xyminmax(box, to_int=False):
    tl, tr, br, bl = box
    x_idx, y_idx = 0, 1
    xmin, xmax, ymin, ymax,  = tl[x_idx], br[x_idx], tl[y_idx], bl[y_idx]
    if to_int:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    return xmin, ymin, xmax, ymax


def batch_box_coordinate_to_xyminmax(boxes, to_int=False):
    xyminmax_boxes = []
    for box in boxes:
        res = box_coordinate_to_xyminmax(box, to_int=to_int)
        res = np.array(res).astype(np.float32)
        xyminmax_boxes.append(res)

    xyminmax_boxes = np.array(xyminmax_boxes).astype(np.float32)
    return xyminmax_boxes


def box_xyminmax_to_coordinates(box, to_int=False):
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


def batch_box_xyminmax_to_coordinates(boxes, to_int=False):
    four_points = []
    for box in boxes:
        res = box_xyminmax_to_coordinates(box, to_int=to_int)
        res = np.array(res).astype(np.float32)
        four_points.append(res)
    return four_points


def box_coordinate_to_xywh(box):
    xmin, ymin, xmax, ymax = box_coordinate_to_xyminmax(box)
    return xmin, ymin, xmax - xmin, ymax - ymin


def box_pad(box, factor=0.1, to_int=True):
    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin
    wf, hf = w * factor, h * factor
    xmin, ymin, xmax, ymax = xmin - wf, ymin - hf, xmax + wf, ymax + hf
    box_out = [xmin, ymin, xmax, ymax]
    if to_int:
        box_out = [int(xmin), int(ymin), int(xmax), int(ymax)]

    #check if minus set to zero
    for idx, val in enumerate(box_out):
        if val < 0: box_out[idx] = 0

    return tuple(box_out)


def box_bounds(box: List[tuple], use_pad=False, pad_factor=0.1) -> List:
    bounds = Polygon(box).bounds
    xmin, ymin, xmax, ymax = bounds
    if use_pad:
        xmin, ymin, xmax, ymax = box_pad(bounds, factor=pad_factor)
    return [int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))]


def box_order(box: List[tuple], use_pad=False, pad_factor=0.1) -> List[tuple]:
    xmin, ymin, xmax, ymax = box_bounds(box, use_pad=use_pad, pad_factor=pad_factor)
    tl, tr, br, bl = (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
    return [tl, tr, br, bl]


def box_centroid(box: List[tuple]) -> tuple:
    pbox = Polygon(box).centroid
    return int(pbox.x), int(pbox.y)


def box_delta_xy(box: List[tuple], use_pad=False, pad_factor=0.1):
    xmin, ymin, xmax, ymax = box_bounds(box, use_pad=use_pad, pad_factor=pad_factor)
    dx = xmax - xmin
    dy = ymax - ymin
    return dx, dy


def triangle_centroid(box_center, bp_left, bp_right) -> tuple:
    tcenter = Polygon([box_center, bp_left, bp_right]).centroid
    return int(tcenter.x), int(tcenter.y)


def box_center_triangle_top_bottom(box: List[tuple]) -> tuple:
    box_tl, box_tr, box_br, box_bl = box
    box_center = box_centroid(box)
    tct = triangle_centroid(box_center, box_tl, box_tr)
    tcb = triangle_centroid(box_center, box_bl, box_br)
    return tct, tcb


def affine_box(fbox: List[tuple], sbox: List[tuple], use_pad: bool = False, pad_factor: float = 0.2) -> List[tuple]:
    # fbox_tl, fbox_tr, fbox_br, fbox_bl = fbox
    # fbox_center = box_centroid(fbox)
    # fbox_tct = triangle_centroid(fbox_center, fbox_tl, fbox_tr)
    # fbox_tcb = triangle_centroid(fbox_center, fbox_bl, fbox_br)
    #
    # sbox_tl, sbox_tr, sbox_br, sbox_bl = sbox
    # sbox_center = box_centroid(sbox)
    # sbox_tct = triangle_centroid(sbox_center, sbox_tl, sbox_tr)
    # sbox_tcb = triangle_centroid(sbox_center, sbox_bl, sbox_br)

    fbox_tct, fbox_tcb = box_center_triangle_top_bottom(fbox)
    sbox_tct, sbox_tcb = box_center_triangle_top_bottom(sbox)

    tl, tr = fbox_tct, sbox_tct
    br, bl = sbox_tcb, fbox_tcb
    afbox = [tl, tr, br, bl]

    # afbox_tl, afbox_tr, afbox_br, afbox_bl = fbox_tct, sbox_tct, sbox_tcb, fbox_tcb
    # afbox = [afbox_tl, afbox_tr, afbox_br, afbox_bl]
    afbox = box_order(afbox, use_pad=use_pad, pad_factor=pad_factor)
    return afbox


def affine_boxes(boxes: List[List], use_pad: bool = False, pad_factor: float = 0.2) -> List[List]:
    out = []
    c = len(boxes)
    for i in range(c):
        if (i + 1) < c:
            fbox, sbox = boxes[i], boxes[i + 1]
            aff_box = affine_box(fbox, sbox, use_pad=use_pad, pad_factor=pad_factor)
            out.append(aff_box)
    return out


def iou(boxa: List, boxb: List) -> float:
    a = Polygon(boxa)
    b = Polygon(boxb)
    intersect_area = a.intersection(b).area
    union_area = a.union(b).area
    return intersect_area / union_area


def intersect_over_union(boxa: List, boxb: List, threshold: float = 0.1) -> bool:
    iou_result = iou(boxa, boxb)
    if iou_result > threshold:
        # print(iou_result)
        return True
    return False


def find_word_over_character_box(word_bbox: List, char_bbox: List, threshold: float = 0.05) -> List:
    word_bbox = sorted(word_bbox)
    char_bbox = sorted(char_bbox)
    wboxes = []
    for i in range(len(word_bbox)):
        word = []
        for j in range(len(char_bbox)):
            iou = intersect_over_union(word_bbox[i], char_bbox[j], threshold=threshold)
            if iou:
                word.append(char_bbox[j])
        wboxes.append(word)
    return wboxes


def affinity_boxes(wbbox: List, cbbox: List, threshold: float = 0.05,
                   use_pad: bool = False, pad_factor: float = 0.2) -> List:
    if type(wbbox) is not list:
        wbbox = wbbox.tolist()
        wbbox = sorted(wbbox)
    if type(cbbox) is not list:
        cbbox = cbbox.tolist()
        cbbox = sorted(cbbox)

    wboxes = find_word_over_character_box(wbbox, cbbox, threshold=threshold)
    afboxes = []
    for wb in wboxes:
        afb = affine_boxes(wb, use_pad=use_pad, pad_factor=pad_factor)
        afboxes = afboxes + afb
    return afboxes


if __name__ == '__main__':
    pass
    # box1 = [(100, 100), (150, 100), (100, 150), (150, 150)]
    # box2 = [(153, 110), (153, 140), (200, 110), (200, 140)]
    # box3 = [(203, 145), (203, 145), (250, 105), (210, 105)]
    # boxes = [box1, box2, box3]
    # boxes_ordered = []
    # for box in boxes:
    #     boxes_ordered.append(box_order(box))
    # boxes_ordered
    #
    # region_score = gaussian_generator(boxes=boxes_ordered, image_size=(300, 300))
    #
    # afboxes = affinity_boxes(boxes_ordered)
    # affinity_score = gaussian_generator(boxes=afboxes, image_size=(300, 300))
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(5, 5))
    # plt.imshow(region_score);
    # plt.show()
    # plt.figure(figsize=(5, 5))
    # plt.imshow(affinity_score);
    # plt.show()
