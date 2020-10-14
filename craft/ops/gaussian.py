import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# from craft_iqra.utils import box_utils
from . import boxes as box_utils
from typing import *


def _isotropic_gaussian_heatmap(dratio: float = 3.0, ksize: tuple = (50,50), to_color: bool = False):
    kx, ky = ksize
    scaled_gaussian = lambda x: math.exp(-(1 / 2) * (x ** 2))
    gaussian2d_heatmap = np.zeros((kx, ky), np.uint8)
    for i in range(kx):
        for j in range(ky):
            distance_from_center = np.linalg.norm(np.array([i - kx / 2, j - ky / 2]))
            distance_from_center = dratio * distance_from_center / (max(kx, ky) / 2)
            scaled_gaussian_prob = scaled_gaussian(distance_from_center)
            gaussian2d_heatmap[i, j] = np.clip(scaled_gaussian_prob * 255, 0, 255)

    if to_color:
        gaussian2d_heatmap = cv.applyColorMap(gaussian2d_heatmap, cv.COLORMAP_JET)
        gaussian2d_heatmap = cv.cvtColor(gaussian2d_heatmap, 4)

    return gaussian2d_heatmap


def _perspective_transform(image: np.ndarray, box):
    h, w = image.shape[0], image.shape[1]
    dx, dy = box_utils.delta_xy(box)
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    dst_pts = np.array([[0, 0], [dx, 0], [dx, dy], [0, dy]], dtype="float32")
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(image, M, (dx, dy), flags=cv.INTER_CUBIC)
    return warped


class GaussianGenerator(object):
    def __init__(self, ksize: Tuple = (50, 50), dratio: float = 3, 
                 use_pad: bool = False, pad_factor: float = 0.1, to_color: bool = False):
        self.ksize = ksize
        self.dratio = dratio
        self.to_color = to_color
        self.use_pad = use_pad
        self.pad_factor = pad_factor
        self.gaussian2d = _isotropic_gaussian_heatmap(dratio=dratio, ksize=ksize, to_color=to_color)

    def __call__(self, boxes, image_size: Tuple):
        boxes = sorted(boxes)
        image = np.zeros(image_size)
        g2dheatmap = self.gaussian2d.copy()
        for box in boxes:
            box = box_utils.order(box, use_pad=self.use_pad, pad_factor=self.pad_factor)
            xmin, ymin, xmax, ymax = box_utils.bounds(box)

            if xmin <= 0: xmin, xmax = abs(xmin), xmax + abs(xmin)
            if ymin <= 0: ymin, ymax = abs(ymin), ymax + abs(ymin)

            warped = _perspective_transform(g2dheatmap, box)
#             print(warped.shape)
            cropped = image[ymin:ymax, xmin:xmax]
            ch, cw = cropped.shape
            nsize = (cw, ch)
            warped = cv.resize(warped, nsize, interpolation=cv.INTER_AREA)
            image[ymin:ymax, xmin:xmax] = np.add(warped, cropped)

        return image


if __name__ == '__main__':
    g = GaussianGenerator()
    plt.imshow(g.gaussian2d_heatmap_image)
