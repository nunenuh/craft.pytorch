import cv2 as cv
import numpy as np
import torch
from skimage import io
from torchvision.transforms import functional as F

from ..utils import craft_utils, imgproc
from ..utils import box_utils


def load_image(path):
    img = io.imread(path)  # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:, :, :3]
    img = np.array(img, dtype=np.uint8)

    return img


def resize_aspect_ratio(img, square_size=1280, interpolation=cv.INTER_LINEAR, mag_ratio=1.5):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, square_size=square_size,
                                                                          interpolation=interpolation,
                                                                          mag_ratio=mag_ratio)
    return img_resized, target_ratio, size_heatmap


def resize(img: np.ndarray, dsize=(768, 768), interpolation='linear'):
    img_resized = cv.resize(img, dsize=dsize, interpolation=interpolation)
    return img_resized


def normalize_variances(img: torch.Tensor, mean: list = [0.485, 0.456, 0.406],
                        std: list = [0.229, 0.224, 0.225],
                        inplace: bool = False):
    return F.normalize(img, mean, std, inplace)


def normalize_dim(img: torch.Tensor):
    if len(img.size()) == 3:
        return img.unsqueeze(dim=0)
    else:
        raise Exception(f"image dimension is {img.size()} and has length of {len(img.size())}, "
                        f"it has to be 3 dim to continue this process!")


def load_image_tensor(path, dsize=1280, mag_ratio=1.5, device='cpu'):
    img = load_image(path)
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, square_size=dsize, mag_ratio=mag_ratio)
    img_tensor = F.to_tensor(img_resized)
    img_normalized = normalize_variances(img_tensor)
    img_norm_dim = normalize_dim(img_normalized)
    img_device = img_norm_dim.to(device)
    return img_device, target_ratio, size_heatmap, img_resized


def minmax_scale(npar: np.ndarray) -> np.ndarray:
    return (npar - npar.min()) / (npar.max() - npar.min())


def tensor_minmax_scale(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def from_tensor_to_numpy(tensor: torch.Tensor, squeeze=False, permute=False, to_uint8=False):
    if permute: tensor = tensor.permute(1, 2, 0)
    if squeeze: tensor = tensor.squeeze()
    if tensor.requires_grad: tensor = tensor.detach()
    if tensor.is_cuda:
        tnp = tensor.cpu().numpy()
    else:
        tnp = tensor.numpy()
    if to_uint8: tnp = tnp.astype(np.uint8)
    return tnp


def revert_back(tensor, size=(768, 768), squeeze=False, permute=False, to_uint8=False):
    tnp = from_tensor_to_numpy(tensor, squeeze=squeeze, permute=permute, to_uint8=to_uint8)
    return cv.resize(tnp, dsize=size, interpolation=cv.INTER_AREA)


def word_boxes(img, reg, aff, text_threshold=0.7, link_threshold=0.3, low_text=0.3, poly=False, mag_ratio=1):
    height, width, channel = img.shape
    target_size = mag_ratio * max(height, width)
    target_ratio = target_size / max(height, width)
    ratio_h = ratio_w = 1 / target_ratio

    boxes, polys = craft_utils.getDetBoxes(reg, aff, text_threshold, link_threshold, low_text, poly)
    return boxes, polys


def boxes_to_images(img, boxes, use_pad=True, pad_factor=0.05):
    images_patch = []
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32)
        bbox = box_utils.box_bounds(poly, use_pad=use_pad, pad_factor=pad_factor)
        xmin, ymin, xmax, ymax = bbox
        img_patch = img[ymin:ymax, xmin:xmax]
        images_patch.append(img_patch)
    return images_patch


def char_bbox(score, use_pad=True, pad=0.19, low_text=0.3):
    text_score = score.copy()
    boxes = []

    ret, text_score = cv.threshold(text_score, low_text, 1, 0)
    text_score = (text_score * 255).astype(np.uint8)

    cnts = cv.findContours(text_score, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda ctr: cv.boundingRect(ctr)[0] + cv.boundingRect(ctr)[1] * score.shape[1])
    # cnts.sort()
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        if use_pad:
            box = x, y, x + w, y + h
            xmin, ymin, xmax, ymax = box_utils.box_pad(box, pad)
        else:
            box = x, y, x + w, y + h
            xmin, ymin, xmax, ymax = box

        # tl, tr, br, bl = box_utils.box_order(box)
        tl, tr, br, bl = (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
        #
        boxes.append(np.array([tl, tr, br, bl]).astype(np.float32))
    # # boxes = np.sort(np.array(boxes).astype(np.float32), axis=0)
    # # boxes = [boxes.squeeze()]
    # boxes = sorted(boxes, key=lambda x:len(x))

    # print(boxes)

    return boxes


# change from four coordinat to xyminmax
def sort_boxes_lrtb(boxes, pad=10):
    xmin_index, ymin_index, xmax_index, ymax_index = 0, 1, 2, 3
    boxes_xymm = box_utils.batch_box_coordinate_to_xyminmax(boxes)
    boxes_sorted_xymin = sorted(boxes_xymm.tolist(), key=lambda box: box[ymin_index])

    lymin = np.min(boxes_xymm[:, ymin_index])  # sum minus of ymin
    lymax = np.min(boxes_xymm[:, ymax_index])
    box_line, boxes_line = [], []
    for idx, box in enumerate(boxes_sorted_xymin):
        xmin, ymin, xmax, ymax = box

        if ymax <= lymax:
            lymin = ymin
            box_line.append(box)
        else:
            box_line = sorted(box_line, key=lambda box: box[xmin_index])
            # print(box_line)
            # box_line = sorted(box_line, key=lambda box: box[ymin_index])

            boxes_line = boxes_line + box_line

            box_line = []  # reset box line
            box_line.append(box)
        thval = (ymax - lymin) + pad
        lymax = thval + lymin

    # assert len(boxes)==len(boxes_line), f"len of boxes ({len(boxes)}) in is not the same with box out ({len(boxes_line)})!"

    boxes = box_utils.batch_box_xyminmax_to_coordinates(boxes_line)

    return boxes


def sort_boxes_lrtb_segmented(boxes, pad=10):
    xmin_index, ymin_index, xmax_index, ymax_index = 0, 1, 2, 3
    boxes_xymm = box_utils.batch_box_coordinate_to_xyminmax(boxes)
    boxes_sorted_xymin = sorted(boxes_xymm.tolist(), key=lambda box: box[ymin_index])

    lymin = np.min(boxes_xymm[:, ymin_index])  # sum minus of ymin
    lymax = np.min(boxes_xymm[:, ymax_index])
    box_line, boxes_line = [], []
    for idx, box in enumerate(boxes_sorted_xymin):
        xmin, ymin, xmax, ymax = box

        if ymax <= lymax:
            lymin = ymin
            box_line.append(box)
        else:
            box_line = sorted(box_line, key=lambda box: box[xmin_index])
            box_line = box_utils.batch_box_xyminmax_to_coordinates(box_line)
            boxes_line.append(box_line)

            box_line = []  # reset box line
            box_line.append(box)
        thval = (ymax - lymin) + pad
        lymax = thval + lymin

    return boxes_line
