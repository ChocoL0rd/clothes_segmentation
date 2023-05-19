from torch.utils.data import Dataset
from torchvision.transforms.transforms import ToTensor
import torch
import torch.nn.functional as F
import random

import albumentations as A

import numpy as np

from PIL import Image
import cv2 as cv

from typing import Dict, Optional, Union, List
import os
import json

import logging

from .aug_tools import *

__all__ = [
    "cfg2datasets",
    "ImgMaskSet",
    "MultiImgMaskSet"
]

log = logging.getLogger(__name__)
img2tensor = ToTensor()


def path2img(img_path):
    img = cv.imread(img_path)

    if img is None:
        msg = f"Wrong reading image {img_path}"
        log.critical(msg)
        raise Exception(msg)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # convert to RGB format

    return img


def path2img_mask(img_path, mask_path):
    # read img and mask
    img = path2img(img_path)

    with Image.open(mask_path) as mask_im:
        mask = np.array(mask_im.split()[-1]) / 255
        mask = mask.reshape([mask.shape[0], mask.shape[1], 1])

    return img, mask


class ImgMaskSet(Dataset):
    """
    It's dataset that returns: image name, image tensor, mask tensor, (original image, original mask if needed)
    In img and mask dirs imgs and corresponding masks should be named the same.
    fgr and bgr trfms have not to change mask (it's can not be a flip, for instance)
    """
    def __init__(self, log_name: str, img_dir_path: str, mask_dir_path: str, img_list: List[str],
                 max_size: int,
                 bgr_trfm, fgr_trfm, trfm, preproc,
                 device: torch.device,):
        """
        :param log_name:  name that is used in log

        :param img_dir_path: path to directory where images are contained. in this directory all images are .jpg
        :param mask_dir_path: path to directory where masks (images with deleted background) are contained.
        in this directory all images are .png
        :param img_list: specifies image names in image directory that should be used (without extension)

        :param bgr_trfm: transformation of background, is not used during the test
        :param fgr_trfm: foreground augmentations, is not used during the test
        :param trfm: transformations to augment dataset, is not used during the test
        :param preproc: transformations that used during the test, it is applied after all other transformations

        :param device: device of images and masks
        """

        self.log_name = log_name

        self.img_dir_path = img_dir_path
        self.mask_dir_path = mask_dir_path
        self.device = device

        self.img_list = img_list.copy()
        self.tmp_img_list = img_list.copy()

        if max_size == -1 or max_size >= len(self.img_list):
            self.size_is_bounded = False
            self.size = len(self.img_list)
        else:
            self.size_is_bounded = True
            self.size = max_size

        # augmentation
        self.bgr_trfm = bgr_trfm
        self.fgr_trfm = fgr_trfm
        self.trfm = trfm
        self.aug_flag = True  # applying of augmentation depends on it

        # preprocessing
        self.preproc = preproc
        self.preproc_flag = True  # applying of preprocessing depends on it

        # needed to return img in it original form, without preproc and augmentation
        self.return_original = False

        log.info(f"Created {self.log_name} ImgMaskSet: \n"
                 f"Size: {self.size} \n"
                 f"Real size: {len(self.img_list)} \n"
                 f"Device: {self.device} \n")

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):

        if self.size_is_bounded:
            if len(self.tmp_img_list) == 0:
                self.tmp_img_list = self.img_list.copy()
            idx = random.randint(0, len(self.tmp_img_list)-1)
            self.tmp_img_list.pop(idx)

        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir_path, img_name + ".jpg")
        mask_path = os.path.join(self.mask_dir_path, img_name + ".png")
        original_img, original_mask = path2img_mask(img_path, mask_path)

        img = original_img.copy()
        mask = original_mask.copy()

        # apply transformations
        if self.aug_flag:
            img, mask = self.apply_aug(img, mask)

        if self.preproc_flag:
            img, mask = self.apply_preproc(img, mask)

        # convert to tensor and transfer to device
        img_tensor = img2tensor(img).to(torch.float32).to(self.device)
        mask_tensor = img2tensor(mask).to(torch.float32).to(self.device)

        # return original image if it's needed
        if self.return_original:
            original_img = img2tensor(original_img).to(self.device)
            return img_name, img_tensor, mask_tensor, original_img, original_mask
        else:
            return img_name, img_tensor, mask_tensor

    def apply_aug(self, img, mask):
        trfmd_bgr = self.bgr_trfm(image=img, mask=mask)["image"]
        trfmd_fgr = self.fgr_trfm(image=img, mask=mask)["image"]
        img = (trfmd_fgr * mask + trfmd_bgr * (1 - mask)).astype("uint8")

        augmented = self.trfm(image=img, mask=mask)
        return augmented["image"], augmented["mask"]

    def apply_preproc(self, img, mask):
        preprocessed = self.preproc(image=img, mask=mask)
        return preprocessed["image"], preprocessed["mask"]

    def set_aug_flag(self, value: bool):
        self.aug_flag = value

    def set_preproc_flag(self, value: bool):
        self.preproc_flag = value

    def set_return_original(self, value: bool):
        self.return_original = value

    def get_img_list(self):
        return self.img_list


class MultiImgMaskSet(Dataset):
    """
    Returns  directory name, image name, image tensor, mask tensor, (original image, original mask if needed)
    """
    def __init__(self, dir_dict: dict, log_name: str, root_path: str, max_subset_size: int,
                 bgr_trfm, fgr_trfm, trfm, preproc,  # add type of augmentation
                 device: torch.device):
        """
        :param dir_dict: {dir_name: img_list} (for each img png and jpg analogue in that list)
        :param log_name: name of the dataset
        :param root_path: path to the dir that contains dirs with images.
        :param bgr_trfm: check in ImgMaskSet
        :param fgr_trfm: check in ImgMaskSet
        :param trfm: check in ImgMaskSet
        :param preproc: check in ImgMaskSet
        :param device: check in ImgMaskSet
        """

        self.log_name = log_name
        self.root_path = root_path
        self.device = device

        self.dataset_names = [dir_name for dir_name in dir_dict.keys()]
        self.img_dict = {}  # {dir_name: img_list}
        for dir_name, dir_list in dir_dict.items():
            self.img_dict[dir_name] = [img_name for img_name in dir_dict[dir_name]]

        self.datasets_dict = {}  # {dir_name: dataset}
        for dir_name in self.dataset_names:
            # each datasets name is a dir name it contains
            self.datasets_dict[dir_name] = ImgMaskSet(log_name=f"[{self.log_name}][{dir_name}]",
                                                      img_dir_path=os.path.join(self.root_path, dir_name),
                                                      mask_dir_path=os.path.join(self.root_path, dir_name),
                                                      img_list=self.img_dict[dir_name],
                                                      max_size=max_subset_size,
                                                      bgr_trfm=bgr_trfm,
                                                      fgr_trfm=fgr_trfm,
                                                      trfm=trfm,
                                                      preproc=preproc,
                                                      device=self.device
                                                      )

        # sizes of each dataset with index corresponding to it index in dataset_names
        self.sizes = [self.datasets_dict[name].size for name in self.dataset_names]

        # sum of sizes of all datasets
        self.size = sum(self.sizes)

        self.aug_flag = True  # applying of augmentation depends on it
        self.preproc_flag = True  # applying of preprocessing depends on it
        # needed to return img in it original form, without preproc and augmentation
        self.return_original = False

        log.info(f"Created {self.log_name} MultiImgMaskSet: \n"
                 f"Size: {self.size} \n"
                 f"Real size: {sum([len(self.datasets_dict[name].img_list) for name in self.dataset_names])} \n"
                 f"Device: {self.device} \n")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        for i, name in enumerate(self.dataset_names):
            dataset = self.datasets_dict[name]
            size = self.sizes[i]
            if idx >= size:
                idx -= size
            else:
                return name, *dataset[idx]

    def set_aug_flag(self, value: bool):
        self.aug_flag = value
        for dir_name in self.dataset_names:
            self.datasets_dict[dir_name].set_aug_flag(value)

    def set_preproc_flag(self, value: bool):
        self.preproc_flag = value
        for dir_name in self.dataset_names:
            self.datasets_dict[dir_name].set_preproc_flag(value)

    def set_return_original(self, value: bool):
        self.return_original = value
        for dir_name in self.dataset_names:
            self.datasets_dict[dir_name].set_return_original(value)

    def dict2file(self, path: str):
        with open(os.path.join(path, f"{self.log_name}_dataset.json"), "w") as fp:
            json.dump(
                self.img_dict,
                fp, indent=4)


def cfg2datasets(cfg):
    """
    :param cfg: dataset_cfg from main config
    consist of:
        1) device - where to contain returned images
        2) path - path to the file to read to get list of images for each dataset
        and path to the folder where it contains.
        3) filter - manipulations with datasets to get new
        4) bgr_trfm, fgr_trfm, trfm, preproc

    :return: dictionary: {dataset_name1: dataset1, ...}
    """
    with open(cfg["path"], "r") as f:
        file_dict = json.load(f)

    root_path = os.path.join(os.path.dirname(cfg["path"]), file_dict["root_dir"])

    # converting configs to transformations
    bgr_trfm = cfg2trfm(cfg["bgr_trfm"])
    fgr_trfm = cfg2trfm(cfg["fgr_trfm"])
    trfm = cfg2trfm(cfg["trfm"])
    preproc = cfg2trfm(cfg["preproc"])

    # creating
    datasets = {}
    for dataset_name, dir_dict in file_dict["datasets"].items():
        if dataset_name == "test":
            datasets[dataset_name] = MultiImgMaskSet(
                log_name=dataset_name, root_path=root_path,
                dir_dict=dir_dict, max_subset_size=-1,
                bgr_trfm=A.Compose([]), fgr_trfm=A.Compose([]), trfm=A.Compose([]), preproc=preproc,
                device=torch.device(cfg["device"])
            )
        else:
            datasets[dataset_name] = MultiImgMaskSet(
                log_name=dataset_name, root_path=root_path,
                dir_dict=dir_dict, max_subset_size=cfg["max_subset_size"],
                bgr_trfm=bgr_trfm, fgr_trfm=fgr_trfm, trfm=trfm, preproc=preproc,
                device=torch.device(cfg["device"])
            )

    return datasets

