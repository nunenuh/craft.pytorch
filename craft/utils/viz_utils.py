import cv2 as cv
import numpy as np
import torch

from craft_iqra.craft_source import craft_utils
from craft_iqra.utils import box_utils

sub_title = [
    'Ori Image', 'Ori Region', 'Ori Affinity', 'Ori Reg & Aff',
    'Ori Full BBOX', 'Image Region Overlay', 'Image Affinity Overlay', 'Image RegAff Overlay',
    'Calc Image Word BBOX', 'Calc Image Char BBOX', 'Original Char BBOX', 'Original Word BBOX'
]


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


def draw_rect(img, charbb, color=(0, 255, 0), thick=1):
    img = img.copy()
    for bbox in charbb:
        xmin, ymin, xmax, ymax = box_utils.box_bounds(bbox)
        img = cv.rectangle(img, (xmin, ymin), (xmax, ymax), color, thick)
    return img


def word_bbox_draw_rect(img, reg, aff, color=(0, 0, 255), thick=1,
                        text_threshold=0.7, link_threshold=0.3, low_text=0.3, poly=False, mag_ratio=1):
    height, width, channel = img.shape
    target_size = mag_ratio * max(height, width)
    target_ratio = target_size / max(height, width)
    ratio_h = ratio_w = 1 / target_ratio

    boxes, polys = craft_utils.getDetBoxes(reg, aff, text_threshold, link_threshold, low_text, poly)
    render_img = reg.copy()
    img_cp = draw_rect(img.copy(), boxes, color=color, thick=thick)
    return img_cp


def find_bbox_and_draw_rect(image, score, use_pad=True, pad=0.19, low_text=0.3, color=(0, 255, 0), thick=1):
    img, original, text_score = image.copy(), image.copy(), score.copy()
    roi_number, roi_images, boxes = 0, [], []

    ret, text_score = cv.threshold(text_score, low_text, 1, 0)
    text_score = (text_score * 255).astype('uint8')

    cnts = cv.findContours(text_score, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        if use_pad:
            box = x, y, x + w, y + h
            xmin, ymin, xmax, ymax = box_utils.box_pad(box, pad)
            x, y, w, h = int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)

        boxes.append([x, y, w, h])
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x + w, y + h), color, thick)
        roi = original[y:y + h, x:x + w]
        roi_images.append(roi)
        roi_number += 1

    return img, roi_images, boxes


def to_colormapjet(image: np.ndarray) -> np.ndarray:
    image = cv.applyColorMap(image, cv.COLORMAP_JET)
    return cv.cvtColor(image, cv.COLOR_RGB2BGR)


def image_overlay(image: np.ndarray, overlay: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    beta = 1.0 - alpha
    return cv.addWeighted(overlay, alpha, image, beta, 0)


def minmax_scale(npar: np.ndarray) -> np.ndarray:
    return (npar - npar.min()) / (npar.max() - npar.min())


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.min() == 0 and image.max() == 1:
        return (image * 255).astype(np.uint8)
    else:
        raise Exception("Image is not in minmax_scale, make sure your image min 0.0 and max is 1.0!")

#
# if __name__ == '__main__':
#     weights_trained = torch.load('../../weights/ktp_ohem_best.pth.tar', map_location=torch.device('cpu'))
#     print(weights_trained['best_lost'], weights_trained['epoch'])
#     weights_trained = copy_state_dict(weights_trained['state_dict'])
#
#     model_trained = CRAFT(pretrained=True)
#     model_trained.load_state_dict(weights_trained)
#
#     tr = tc.Compose([
#         tc.Resize(size=(768, 768)),
#         #     tc.RandomRotation(degrees=15),
#         tc.RandomCrop(size=(768, 768)),
#         # tc.RandomVerticalFlip(),
#         #     tc.RandomHorizontalFLip(),
#         #     tc.RandomContrast(),
#         #     tc.RandomBrightness(),
#         #     tc.RandomColor(),
#         #     tc.RandomSharpness(),
#         tc.ScaleRegionAffinity(scale=0.5),
#         tc.NumpyToTensor()
#     ])
#
#     path = '/data/tigapilar/ktp/processed/clean/ready'
#     trainset = TigapilarDataset(path, mode='train', transform=tr,
#                                 gauss_ksize=(35, 35), gauss_dratio=2.0,
#                                 gauss_use_pad=False, gauss_pad_factor=0.05, aff_thresh=0.01, )
#     validset = TigapilarDataset(path, mode='valid', transform=tr,
#                                 gauss_ksize=(35, 35), gauss_dratio=2.0,
#                                 gauss_use_pad=False, gauss_pad_factor=0.05, aff_thresh=0.01, )
#
#     len(trainset), len(validset)
#
#     idx = 20
#     impath, img, t_img, reg, t_reg, aff, t_aff, charbb, wordbb = validset.get(idx)
#     data_ori = img, t_img, reg, t_reg, aff, t_aff, charbb, wordbb
#
#     score, _ = model_trained(t_img.unsqueeze(dim=0))
#     score = score.squeeze()
#     p_reg = score[:, :, 0]
#     p_aff = score[:, :, 1]
#
#     t_img = revert_back(t_img, size=(img.shape[1], img.shape[0]), permute=True)
#     p_reg = revert_back(p_reg, size=(img.shape[1], img.shape[0]), squeeze=True)
#     p_aff = revert_back(p_aff, size=(img.shape[1], img.shape[0]), squeeze=True)
#
#     data_pred = img, t_img, p_reg, p_reg, p_aff, p_aff, charbb, wordbb
#
#     sub_title_pred = [
#         'Ori Image', 'Ori Region', 'Ori Affinity', 'Ori Reg & Aff',
#         'Ori Full BBOX', 'Image Region Overlay', 'Image Affinity Overlay', 'Image RegAff Overlay',
#         'Calc Image Word BBOX', 'Calc Image Char BBOX', 'Original Char BBOX', 'Original Word BBOX'
#     ]
#
#     sub_title_ori = [
#         'Ori Image', 'Ori Region', 'Ori Affinity', 'Ori Reg & Aff',
#         'Ori Full BBOX', 'Image Region Overlay', 'Image Affinity Overlay', 'Image RegAff Overlay',
#         'Calc Image Word BBOX', 'Calc Image Char BBOX', 'Original Char BBOX', 'Original Word BBOX'
#     ]
#
#     oripath = 'result' + impath.stem + '_gt.png'
#     predpath = 'result' + impath.stem + '_pred.png'
#
#     visual_analysis(data_ori, path=oripath, title='Ground Truth of File ' + impath.name,
#                     subplot_title=sub_title_ori)
#     visual_analysis(data_pred, path=predpath, title='Prediction of File ' + impath.name,
#                     subplot_title=sub_title_pred)
