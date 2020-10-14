import cv2
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from shapely.geometry import Polygon

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from ..ops import boxes, affinity
from ..ops.gaussian import GaussianGenerator
from . import utils


class SynthTextDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, split_val=0.2, percent_usage=100,
                 gauss_ksize=(50, 50), gauss_dratio=3.0, aff_thresh=0.05,
                 gauss_use_pad=False, gauss_pad_factor=0.1):
        self.root = Path(root)
        self.mode = mode
        self.split_val = split_val
        self.transform = transform
        self.percent_usage = percent_usage
        self.gauss_ksize = gauss_ksize
        self.gauss_dratio = gauss_dratio
        self.aff_threshold = aff_thresh

        self.gaussian = GaussianGenerator(ksize=gauss_ksize, dratio=gauss_dratio,
                                          use_pad=gauss_use_pad, pad_factor=gauss_pad_factor)

        self._log_print(
            "loading file gt.mat, please wait, it will take a while...")
        self.gt_base = loadmat(str(self.root.joinpath("gt.mat")))
        self._log_print("file gt.mat has been loaded...")
        self._build_gt()

    def _split_index_generator(self, ldata):
        vlen = int(self.split_val * ldata)
        t_start, t_end = 0, ldata - vlen
        v_start, v_end = ldata - vlen, ldata
        return t_start, t_end, v_start, v_end

    def _build_gt(self):
        char, word = self.gt_base['charBB'][0], self.gt_base['wordBB'][0]
        imnames, txt = self.gt_base['imnames'][0], self.gt_base['txt'][0]
        ldata = len(imnames)
        t_start, t_end, v_start, v_end = self._split_index_generator(ldata)
        if self.mode == 'train':
            t_index = self._get_sampled_index(t_start, t_end)
            self.gt = {
                'charBB': char[t_index],
                'wordBB': word[t_index],
                'imnames': imnames[t_index],
                'txt': txt[t_index]
            }
        else:
            v_index = self._get_sampled_index(v_start, v_end)
            self.gt = {
                'charBB': char[v_index],
                'wordBB': word[v_index],
                'imnames': imnames[v_index],
                'txt': txt[v_index]
            }
        del char, word, imnames, txt
        del self.gt_base

    def _clear_dict(self, dct):
        keys = dct.keys()
        for k in keys:
            del dct[k]

    def _get_sampled_index(self, start, end):
        np.random.seed(1261)
        percent = self.percent_usage / 100
        size = int((end - start) * percent)
        index = np.arange(start, end)
        new_index = np.random.choice(index, size, replace=False)
        print(f'log:\tnew size {size} in percentage of {self.percent_usage}% ')

        return new_index

    def __len__(self):
        return len(self.gt['imnames'])

    def _log_print(self, txt):
        print(f'log:\t{txt}')

    def get_raw(self, idx):
        image = self._get_image(idx)
        char = self.gt['charBB'][idx]
        word = self.gt['wordBB'][idx]
        return image, char, word

    def _get_mask(self, shape):
        return np.ones((shape[0], shape[1]), np.float32)

    def _get_region_score_image(self, image: np.ndarray, idx: int):
        h, w = image.shape[0], image.shape[1]
        x, y, tl, tr, br, bl = 0, 1, 0, 1, 2, 3
        char = self.gt['charBB'][idx]
        cx, cy = char[x].T, char[y].T
        boxes = []
        t = len(cx)
        for i in range(t):
            ctl = (cx[i][tl], cy[i][tl])
            ctr = (cx[i][tr], cy[i][tr])
            cbr = (cx[i][br], cy[i][br])
            cbl = (cx[i][bl], cy[i][bl])
            boxes.append([ctl, ctr, cbr, cbl])
        boxes = sorted(boxes)
        region_score = self.gaussian(boxes=boxes, image_size=(h, w))
        return region_score

    def _get_affinity_score_image(self, image: np.ndarray, idx):
        h, w = image.shape[0], image.shape[1]
        char = self.gt['charBB'][idx]
        word = self.gt['wordBB'][idx]
        wbbox = utils.raw2bbox(word, to_int=True)
        cbbox = utils.raw2bbox(char, to_int=True)

        affinity_boxes = affinity.boxes(
            wbbox, cbbox, threshold=self.aff_threshold)
        afscore = self.gaussian(affinity_boxes, image_size=(h, w))
        return afscore, affinity_boxes

    def _get_image(self, idx):
        imname = self.gt['imnames'][idx][0]
        impath = str(self.root.joinpath(imname))
        return utils.load_image(impath)

    def __getitem__(self, idx):
        image = self._get_image(idx)
        regscore = self._get_region_score_image(image, idx)
        afscore, afboxes = self._get_affinity_score_image(image, idx)

        # char = self.gt['charBB'][idx]
        # word = self.gt['wordBB'][idx]
        # wbbox = utils.raw_to_bbox(word, to_int=True)
        # cbbox = utils.raw_to_bbox(char, to_int=True)

        mask = self._get_mask(regscore.shape)
        if self.transform:
            image, regscore, afscore, mask = self.transform(
                image, regscore, afscore, mask)
#         return image, regscore, afscore, mask
        return image, regscore, afscore
        # return image, regscore, afscore, mask, np.array(wbbox), np.array(cbbox), np.array(afboxes)

    def _run_check(self, idx):
        print(f'Run check...')
        print(f'Please wait, this will take several minutes...')
        image_exist = self._check_image_exist(idx)
        if image_exist:
            iou_is_valid = self._check_iou_is_valid(idx)
            if iou_is_valid:
                status = True
            else:
                status = False
                # print(f'IOU is not valid at index {idx}')
        else:
            status = False
            # print(f'Image file is not exist at index {idx}')
        return status

    def _check_iou_is_valid(self, idx):
        charbb = self.gt['charBB'][idx]
        wordbb = self.gt['wordBB'][idx]
        wbbox = utils.raw2bbox(wordbb, to_int=True)
        cbbox = utils.raw2bbox(charbb, to_int=True)
        status = True
        for i in range(len(wbbox)):
            for j in range(len(cbbox)):
                a = Polygon(wbbox[i])
                b = Polygon(cbbox[j])
                if a.is_valid and b.is_valid:
                    intersect_valid = a.intersection(b).is_valid
                    union_valid = a.union(b).is_valid
                    if intersect_valid is False or union_valid is False:
                        status = False
                        break
                else:
                    status = False
                    break
        return status

    def _check_image_exist(self, idx):
        imname = self.gt['imnames'][idx][0]
        impath = self.root.joinpath(imname)
        img = cv2.imread(str(impath))
        if type(img) is type(None):  # image does not exist
            return False
        return True


class SynthTextChecker(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self._log_print(
            "loading file gt.mat, please wait, it will take a while...")
        self.gt_base = loadmat(str(self.root.joinpath("gt.mat")))
        self._log_print("file gt.mat has been loaded...")
        self._build_gt()

    def _log_print(self, txt):
        print(f'log:\t{txt}')

    def _build_gt(self):
        char, word = self.gt_base['charBB'][0], self.gt_base['wordBB'][0]
        imnames, txt = self.gt_base['imnames'][0], self.gt_base['txt'][0]
        ldata = len(imnames)
        self.gt = {
            'charBB': char[0:ldata],
            'wordBB': word[0:ldata],
            'imnames': imnames[0:ldata],
            'txt': txt[0:ldata]
        }

    def run_check(self):
        status = True
        len_data = len(self.gt['imnames'])
        print(f'Run check...')
        print(f'Please wait, this will take several minutes...')
        fail_idx = []
        for idx in range(len_data):
            image_exist = self._check_image_exist(idx)
            if image_exist:
                # bbox_has_minus = self._check_bbox_has_minus(idx)
                # if bbox_has_minus is False:
                iou_is_valid = self._check_iou_is_valid(idx)
                if iou_is_valid:
                    status = True
                else:
                    status = False
                    print(f'IOU is not valid at index {idx}')
                    fail_idx.append(idx)
                # else:
                #     status = False
                #     print(f'BBOX has minus value index {idx}')
            else:
                status = False
                print(f'Image file is not exist at index {idx}')
                fail_idx.append(idx)

            # if status:
            #     print(f'Data with IDX {idx} pass check!')
        return fail_idx

    def _check_image_exist(self, idx):
        imname = self.gt['imnames'][idx][0]
        impath = self.root.joinpath(imname)
        img = cv2.imread(str(impath))
        if type(img) is type(None):  # image does not exist
            return False
        return True

    def _get_pbbox(self, boxes):
        x, y, tl, tr, br, bl = 0, 1, 0, 1, 2, 3
        nboxes = []
        for box in boxes:
            pbox = boxes.bounds(box)
            nboxes.append(pbox)
        return nboxes

    def _check_bbox_has_minus(self, idx):
        charbb = self.gt['charBB'][idx]
        wordbb = self.gt['wordBB'][idx]
        wbbox = self._get_pbbox(utils.raw2bbox(wordbb, to_int=True))
        cbbox = self._get_pbbox(utils.raw2bbox(charbb, to_int=True))

        cstat = True
        for c in cbbox:
            xmin, ymin, xmax, ymax = c
            if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                print(f'{xmin}:{xmax} - {ymin}:{ymax}')
                cstat = False
                break

        wstat = True
        for w in wbbox:
            xmin, ymin, xmax, ymax = w
            if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                print(f'{xmin}:{xmax} - {ymin}:{ymax}')
                wstat = False
                break
        if cstat is True and wstat is True:
            return True
        else:
            return False

    def _check_iou_is_valid(self, idx):
        charbb = self.gt['charBB'][idx]
        wordbb = self.gt['wordBB'][idx]
        wbbox = utils.raw2bbox(wordbb, to_int=True)
        cbbox = utils.raw2bbox(charbb, to_int=True)
        status = True
        for i in range(len(wbbox)):
            for j in range(len(cbbox)):
                a = Polygon(wbbox[i])
                b = Polygon(cbbox[j])
                if a.is_valid and b.is_valid:
                    intersect_valid = a.intersection(b).is_valid
                    union_valid = a.union(b).is_valid
                    if intersect_valid is False or union_valid is False:
                        status = False
                        break
                else:
                    status = False
                    break
        return status


class SynthTextDataLoader(DataLoader):
    def __init__(self):
        pass


class SynthTextSampler(Sampler):
    def __init__(self):
        pass


if __name__ == '__main__':
    st = SynthTextChecker(root='/data/SynthText')
    fail_indexs = st.run_check()
    print(fail_indexs)
