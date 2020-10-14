import numbers
import random
from typing import *

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from . import functional as CF


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: Image, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        if not (mask is None):
            for t in self.transforms:
                img, char, aff, mask = t(img, char, aff, mask)
            return img, char, aff, mask
        else:
            for t in self.transforms:
                img, char, aff = t(img, char, aff)
            return img, char, aff

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RegionAffinityMinMaxScaler(object):
    def __init__(self):
        super(RegionAffinityMinMaxScaler, self).__init__()

    def minmax_scaler(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def __call__(self, img: torch.Tensor, char: torch.Tensor, aff: torch.Tensor,
                 mask: torch.Tensor = None) -> torch.Tensor:
        char = self.minmax_scaler(char)
        aff = self.minmax_scaler(aff)
        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff


class Normalize(object):
    def __init__(self, mean: Tuple or List, std: Tuple or List, inplace: bool = False):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img: torch.Tensor, char: torch.Tensor, aff: torch.Tensor,
                 mask: torch.Tensor = None) -> torch.Tensor:
        img = F.normalize(img, self.mean, self.std, self.inplace)
        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff


class ScaleRegionAffinity(object):
    def __init__(self, scale=0.5, interpolation="linear"):
        super(ScaleRegionAffinity, self).__init__()
        self.scale = scale
        self.interpolation = interpolation

    def _translate(self, inter, lib='pil'):
        if lib == 'pil':
            if inter == "linear":
                return Image.BILINEAR
            elif inter == "cubic":
                return Image.CUBIC
            else:
                return Image.BILINEAR
        elif lib == 'cv2':
            if inter == "linear":
                return cv2.INTER_LINEAR
            elif inter == "cubic":
                return cv2.INTER_CUBIC
            elif inter == "nearest":
                return cv2.INTER_NEAREST
            else:
                return cv2.INTER_LINEAR

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        cv2_inter = self._translate(self.interpolation, lib='cv2')
        h, w = img.shape[0], img.shape[1]
        scale_size = (round(w * self.scale), round(h * self.scale))
        char = cv2.resize(char, dsize=scale_size, interpolation=cv2_inter)
        aff = cv2.resize(aff, dsize=scale_size, interpolation=cv2_inter)
        if not (mask is None):
            mask = cv2.resize(mask, dsize=scale_size, interpolation=cv2_inter)
            return img, char, aff, mask
        return img, char, aff


class Resize(object):
    def __init__(self, size, interpolation="linear"):
        super(Resize, self).__init__()
        self.size = size
        self.interpolation = interpolation

    def _translate(self, inter, lib='pil'):
        if lib == 'pil':
            if inter == "linear":
                return Image.BILINEAR
            elif inter == "cubic":
                return Image.CUBIC
            else:
                return Image.BILINEAR
        elif lib == 'cv2':
            if inter == "linear":
                return cv2.INTER_LINEAR
            elif inter == "cubic":
                return cv2.INTER_CUBIC
            elif inter == "nearest":
                return cv2.INTER_NEAREST
            else:
                return cv2.INTER_LINEAR

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        cv2_inter = self._translate(self.interpolation, lib='cv2')
        img = cv2.resize(img, self.size, interpolation=cv2_inter)
        char = cv2.resize(char, dsize=self.size, interpolation=cv2_inter)
        aff = cv2.resize(aff, dsize=self.size, interpolation=cv2_inter)
        if not (mask is None):
            mask = cv2.resize(mask, dsize=self.size, interpolation=cv2_inter)
            return img, char, aff, mask
        return img, char, aff


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img: np.ndarray, output_size):
        if len(img.shape) == 3:
            h, w, d = img.shape
        elif len(img.shape) == 2:
            h, w = img.shape
        else:
            raise ValueError("img shape is not 3 or 2, it is not an image!")

        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        x = random.randint(0, h - th)
        y = random.randint(0, w - tw)
        return x, y, th, tw

    def _check_size(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray):
        if img.shape[0] == char.shape[0] and img.shape[0] == aff.shape[0]:
            if img.shape[1] == char.shape[1] and img.shape[1] == aff.shape[1]:
                pass
            else:
                raise ValueError("Image has different size!")
        else:
            raise ValueError("Image has different size!")

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        self._check_size(img, char, aff)

        if self.padding is not None:
            img = CF.pad(img, self.padding, self.fill, self.padding_mode)
            char = CF.pad(char, self.padding, self.fill, self.padding_mode)
            aff = CF.pad(aff, self.padding, self.fill, self.padding_mode)
            if not (mask is None):
                mask = CF.pad(mask, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[0] < self.size[1]:
            img = CF.pad(
                img, (self.size[1] - img.shape[0], 0), self.fill, self.padding_mode)
            char = CF.pad(
                char, (self.size[1] - char.shape[0], 0), self.fill, self.padding_mode)
            aff = CF.pad(
                aff, (self.size[1] - aff.shape[0], 0), self.fill, self.padding_mode)
            if not (mask is None):
                mask = CF.pad(
                    mask, (self.size[1] - aff.shape[0], 0), self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and img.shape[1] < self.size[0]:
            img = CF.pad(
                img, (0, self.size[0] - img.shape[1]), self.fill, self.padding_mode)
            char = CF.pad(
                char, (0, self.size[0] - char.shape[1]), self.fill, self.padding_mode)
            aff = CF.pad(
                aff, (0, self.size[0] - aff.shape[1]), self.fill, self.padding_mode)
            if not (mask is None):
                mask = CF.pad(
                    mask, (0, self.size[0] - aff.shape[1]), self.fill, self.padding_mode)

        x, y, h, w = self.get_params(img, self.size)

        img = CF.crop(img, x, y, h, w)
        char = CF.crop(char, x, y, h, w)
        aff = CF.crop(aff, x, y, h, w)

        if not (mask is None):
            mask = CF.crop(mask, x, y, h, w)
            return img, char, aff, mask

        return img, char, aff

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomRotation(object):
    def __init__(self, degrees, center=None, scale=1.0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.center = center
        self.scale = scale

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        angle = self.get_params(self.degrees)
        # print(angle)
        img = CF.rotate(img, angle, self.center, self.scale)
        char = CF.rotate(char, angle, self.center, self.scale)
        aff = CF.rotate(aff, angle, self.center, self.scale)
        if not (mask is None):
            mask = CF.rotate(mask, angle, self.center, self.scale)
            return img, char, aff, mask
        return img, char, aff

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomHorizontalFLip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        if random.random() < self.p:
            img = CF.hflip(img)
            char = CF.hflip(char)
            aff = CF.hflip(aff)
            return img, char, aff
        if not (mask is None):
            mask = CF.hflip(mask)
            return img, char, aff, mask

        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        if random.random() < self.p:
            img = CF.vflip(img)
            char = CF.vflip(char)
            aff = CF.vflip(aff)
            if not (mask is None):
                mask = CF.vflip(mask)
                return img, char, aff, mask
            return img, char, aff

        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomContrast(object):
    def __init__(self, p=0.5, range=(0.5, 2.0)):
        self.p = p
        self.range = range

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        if random.random() < self.p:
            rc = random.uniform(float(self.range[0]), float(self.range[1]))
            img = CF.contrast(img, value=rc)
            if not (mask is None):
                return img, char, aff, mask
            return img, char, aff

        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff


class RandomBrightness(object):
    def __init__(self, p=0.5, range=(0.5, 2.0)):
        self.p = p
        self.range = range

    def __call__(self, img, char, aff, mask: np.ndarray = None):
        if random.random() < self.p:
            rc = random.uniform(float(self.range[0]), float(self.range[1]))
            img = CF.brightness(img, value=rc)
            if not (mask is None):
                return img, char, aff, mask
            return img, char, aff

        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff


class RandomSharpness(object):
    def __init__(self, p: float = 0.5, range: tuple = (0.5, 2.0)):
        self.p = p
        self.range = range

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        if random.random() < self.p:
            rc = random.uniform(float(self.range[0]), float(self.range[1]))
            img = CF.sharpness(img, value=rc)
            if not (mask is None):
                return img, char, aff, mask
            return img, char, aff

        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff


class RandomColor(object):
    def __init__(self, p: float = 0.5, range: Tuple[float, float] = (1.0, 2.0)):
        self.p = p
        self.range = range

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        if random.random() < self.p:
            rc = random.uniform(float(self.range[0]), float(self.range[1]))
            img = CF.color(img, value=rc)
            if not (mask is None):
                return img, char, aff, mask
            return img, char, aff

        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff


class RandomHue(object):
    def __init__(self, p: float = 0.5, range: Tuple[int, int] = (0, 50)):
        self.p = p
        self.range = range

    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        if random.random() < self.p:
            rc = int(random.uniform(self.range[0], self.range[1]))
            img = CF.hue(img, value=rc)
            if not (mask is None):
                return img, char, aff, mask
            return img, char, aff

        if not (mask is None):
            return img, char, aff, mask
        return img, char, aff


class NumpyToTensor(object):
    def __call__(self, img: np.ndarray, char: np.ndarray, aff: np.ndarray, mask: np.ndarray = None):
        img, char, aff, = np.array(img), np.array(char), np.array(aff)
        img = F.to_tensor(img)
        char = F.to_tensor(char)
        aff = F.to_tensor(aff)
        if not (mask is None):
            mask = np.array(mask)
            mask = F.to_tensor(mask)
            return img, char, aff, mask
        else:
            return img, char, aff


if __name__ == '__main__':
    pass
