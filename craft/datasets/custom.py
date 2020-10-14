import json
import random
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

# from craft_iqra import utils
from typing import *

from ..ops import boxes, affinity, synthtext
from ..ops.gaussian import GaussianGenerator
from . import utils


class CustomDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, split_val=0.2, scale_size=0.5,
                 gauss_ksize=(25, 25), gauss_dratio=3.0, gauss_use_pad=False, gauss_pad_factor=0.1,
                 aff_thresh=0.05, image_ext='jpg', **kwargs):
        self.root = Path(root)
        self.mode = mode
        self.split_val = split_val
        self.transform = transform
        self.scale_size = scale_size
        self.image_ext = image_ext
        self.gaussian = GaussianGenerator(ksize=gauss_ksize, dratio=gauss_dratio,
                                          use_pad=gauss_use_pad, pad_factor=gauss_pad_factor)

        self.aff_threshold = aff_thresh
        self.image_files = []
        self.json_files = []
        self._load_files()
        self._split_dataset()


    def _load_files(self):
        base_path = self.root
        index_file = base_path.joinpath('indexmap.csv')
        if not index_file.exists():
            raise FileNotFoundError(
                "indexmap.csv is not found in root directory, make sure the file is exist before we can coninue to load the data!")
        df = pd.read_csv(index_file)

        error_index = []
        for index, data in df.iterrows():
            image = base_path.joinpath(data['image_file'])
            json = base_path.joinpath(data['json_file'])
            if not (image.exists() and json.exists()):
                error_index.append(index)
            else:
                self.image_files.append(image)
                self.json_files.append(json)

        if len(error_index) > 0:
            raise FileNotFoundError(
                f"File in indexmap with number {error_index} is not found, please fix the file before we can continue to load the data!")

    def _split_dataset(self):
        random.seed(1618)
        len_files = len(self.image_files)
        index_list = [i for i in range(len_files)]

        valid_total = int(round(self.split_val * len_files))
        valid_index_list = sorted(random.sample(index_list, valid_total))
        train_index_list = sorted(list(set(index_list).difference(set(valid_index_list))))

        if self.mode == 'train':
            self.image_files = [self.image_files[x] for x in train_index_list]
            self.json_files = [self.json_files[x] for x in train_index_list]
        elif self.mode == 'valid':
            self.image_files = [self.image_files[x] for x in valid_index_list]
            self.json_files = [self.json_files[x] for x in valid_index_list]

    def _get_image(self, idx):
        impath = str(self.image_files[idx])
        return utils.load_image(impath)

    def _load_json_to_dict(self, idx):
        path = str(self.json_files[idx])
        return utils.load_json2dict(path)

    def _extract_json_file(self, idx):
        data = self._load_json_to_dict(idx)
        char_dict = []
        word_dict = []
        for dct in data['object']:
            if dct['class_type'] == 'char':
                char_dict.append(dct)
            elif dct['class_type'] == 'word':
                word_dict.append(dct)
        return data['image'], char_dict, word_dict

    def _convert_dict_to_bbox(self, data_dict):
        values = []
        bbox = []
        for dct in data_dict:
            xmin, xmax = dct['coord_value']['xmin'], dct['coord_value']['xmax']
            ymin, ymax = dct['coord_value']['ymin'], dct['coord_value']['ymax']
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            top_left, top_right = (xmin, ymin), (xmax, ymin)
            bottom_right, bottom_left = (xmax, ymax), (xmin, ymax)
            box = [top_left, top_right, bottom_right, bottom_left]
            bbox.append(box)
            values.append(dct['class_value'])

        bbox = np.array(bbox)
        return bbox, values

    def _get_region_score_image(self, image: np.ndarray, charbb: List):
        h, w = image.shape[0], image.shape[1]
        region_boxes = sorted(charbb.tolist())
        region_score = self.gaussian(boxes=region_boxes, image_size=(h, w))
        return region_score

    def _get_affinity_score_image(self, image: np.ndarray, charbb: List, wordbb: List):
        h, w = image.shape[0], image.shape[1]
        affinity_boxes = affinity.boxes(wordbb, charbb, threshold=self.aff_threshold)
        afscore = self.gaussian(affinity_boxes, image_size=(h, w))
        return afscore, torch.from_numpy(np.array(affinity_boxes))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self._get_image(idx)

        image_dict, char_dict, word_dict = self._extract_json_file(idx)
        charbb, char_val = self._convert_dict_to_bbox(char_dict)
        wordbb, word_val = self._convert_dict_to_bbox(word_dict)

        region_score = self._get_region_score_image(image, charbb)
        affinity_score, affinity_boxes = self._get_affinity_score_image(image, charbb, wordbb)

        if self.transform:
            image, region_score, affinity_score = self.transform(image, region_score, affinity_score)

        charbb, wordbb = torch.Tensor(charbb).permute(2, 1, 0), torch.Tensor(wordbb).permute(2, 1, 0)
        # return image, region_score, affinity_score, charbb, wordbb, affinity_boxes
        # return image, region_score, affinity_score, charbb, char_val, wordbb, word_val
        # print(charbb.shape, wordbb.shape)
        # return image, region_score, affinity_score, charbb, wordbb

        return image, region_score, affinity_score

    def get(self, idx):
        impath = self.image_files[idx]
        image = self._get_image(idx)

        image_dict, char_dict, word_dict = self._extract_json_file(idx)
        charbb, char_val = self._convert_dict_to_bbox(char_dict)
        wordbb, word_val = self._convert_dict_to_bbox(word_dict)

        region_score = self._get_region_score_image(image, charbb)
        affinity_score, affinity_boxes = self._get_affinity_score_image(image, charbb, wordbb)

        if self.transform:
            t_image, t_region_score, t_affinity_score = self.transform(image, region_score, affinity_score)

        # charbb, wordbb = torch.Tensor(charbb), torch.Tensor(wordbb)
        return impath, image, t_image, region_score, t_region_score, affinity_score, t_affinity_score, charbb, wordbb, word_val

    def get_word(self, idx):
        impath = self.image_files[idx]
        image = self._get_image(idx)

        image_dict, char_dict, word_dict = self._extract_json_file(idx)
        wordbb, word_val = self._convert_dict_to_bbox(word_dict)

        return impath, image, wordbb, word_val
