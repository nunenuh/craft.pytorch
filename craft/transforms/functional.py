import sys
import math
import cv2 as cv
import numpy as np
import numbers
import collections
import warnings
from PIL import Image
from PIL import ImageEnhance
import PIL

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def vflip(img: np.ndarray):
    return cv.flip(img, 0)


def hflip(img: np.ndarray):
    return cv.flip(img, 1)


def pad(img: np.ndarray, padding, fill=0, padding_mode='constant'):
    if type(img) is not np.ndarray:
        raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pl = pr = pt = pb = padding
    elif isinstance(padding, Sequence) and len(padding) == 2:
        pl = pr = padding[0]
        pt = pb = padding[1]
    elif isinstance(padding, Sequence) and len(padding) == 4:
        pl, pr, pt, pb = padding[0], padding[1], padding[2], padding[3]

    if len(img.shape) == 3:
        if padding_mode is 'constant':
            img = np.pad(img, ((pt, pb), (pl, pr), (0, 0)), padding_mode, constant_values=fill)
        else:
            img = np.pad(img, ((pt, pb), (pl, pr), (0, 0)), padding_mode)
    # Grayscale image
    if len(img.shape) == 2:
        if padding_mode is 'constant':
            img = np.pad(img, ((pt, pb), (pl, pr)), padding_mode, constant_values=fill)
        else:
            img = np.pad(img, ((pt, pb), (pl, pr)), padding_mode)

    return img


def crop(img: np.ndarray, x: int, y: int, w: int, h: int):
    return img[x:w, y:h]


def hue(img: np.ndarray, value=255):
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    img[:, :, 2] = value  # Changes the V value
    img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    return img


def color(img: np.ndarray, value=1.0):
    img = Image.fromarray(img)
    img = ImageEnhance.Color(img).enhance(value)
    img = np.asarray(img)
    return img


def sharpness(img: np.ndarray, value=1.0):
    img = Image.fromarray(img)
    img = ImageEnhance.Sharpness(img).enhance(value)
    img = np.asarray(img)
    return img


def brightness(img: np.ndarray, value=1.0):
    img = Image.fromarray(img)
    img = ImageEnhance.Brightness(img).enhance(value)
    img = np.asarray(img)
    return img


def contrast(img: np.ndarray, value=1.0):
    img = Image.fromarray(img)
    img = ImageEnhance.Contrast(img).enhance(value)
    img = np.asarray(img)
    return img


def rotate(img: np.ndarray, angle: int, center=None, scale: float = 1.0):
    h, w = img.shape[0], img.shape[1]
    dsize = (w, h)
    if center is None:
        center = (w / 2, h / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    img = cv.warpAffine(img, rotation_matrix, dsize)
    return img


