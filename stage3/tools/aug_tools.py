import cv2

import albumentations as A

from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from albumentations.augmentations.transforms import PixelDropout
from albumentations.augmentations.dropout.channel_dropout import ChannelDropout
from albumentations.augmentations.transforms import ChannelShuffle
from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.augmentations.geometric.transforms import HorizontalFlip
from albumentations.augmentations.geometric.transforms import VerticalFlip
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.blur.transforms import Blur
from albumentations.augmentations.transforms import ToGray
from albumentations.augmentations.transforms import RandomShadow
from albumentations.augmentations.geometric.transforms import GridDistortion
from albumentations.augmentations.geometric.transforms import ElasticTransform
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.transforms import Normalize
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations.augmentations.transforms import RGBShift
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists

import random

import logging

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "cfg2trfm",
]


def cfg2mini_trfm(cfg):
    """
    Convert config to transformation
    """
    name = cfg["name"]
    if name == "custom_aug":
        trfm = A.Compose([
            A.OneOf([
                A.Compose([
                    PixelDropout(dropout_prob=0.01,
                                 per_channel=True,
                                 drop_value=0,
                                 always_apply=True),
                    Blur(always_apply=True)
                ]),
                A.Compose([
                    PixelDropout(dropout_prob=0.005,
                                 per_channel=True,
                                 drop_value=0,
                                 always_apply=True),
                    Blur(always_apply=True)
                ]),
                A.Compose([
                    PixelDropout(dropout_prob=0.0001,
                                 per_channel=True,
                                 drop_value=0,
                                 always_apply=True),
                    Blur(always_apply=True)
                ]),
                PixelDropout(dropout_prob=0.00001,
                             per_channel=True,
                             drop_value=0,
                             p=1),
                PixelDropout(dropout_prob=0.01,
                             per_channel=False,
                             drop_value=0,
                             p=1),
                A.Compose([
                    PixelDropout(dropout_prob=0.01,
                                 per_channel=False,
                                 drop_value=0,
                                 always_apply=True),
                    Blur(always_apply=True)
                ]),
                CoarseDropout(
                    max_holes=4,
                    max_height=50,
                    max_width=50,
                    p=1),

            ]),
        ], p=cfg["p"])
    elif name == "coarse_dropout":
        trfm = CoarseDropout(max_holes=cfg["max_holes"],
                             max_height=cfg["max_height"],
                             max_width=cfg["max_width"],
                             fill_value=cfg["fill_value"],
                             p=cfg["p"])
    elif name == "pixel_dropout":
        trfm = PixelDropout(dropout_prob=cfg["dropout_prob"],
                            per_channel=cfg["per_channel"],
                            drop_value=cfg["drop_value"],
                            p=cfg["p"])
    elif name == "channel_dropout":
        trfm = ChannelDropout(p=cfg["p"])
    elif name == "channel_shuffle":
        trfm = ChannelShuffle(p=cfg["p"])
    elif name == "random_crop":
        trfm = RandomCrop(height=cfg["height"],
                          width=cfg["width"],
                          always_apply=cfg["always_apply"],
                          p=cfg["p"])
    elif name == "vertical_flip":
        trfm = VerticalFlip(p=cfg["p"])
    elif name == "horizontal_flip":
        trfm = HorizontalFlip(p=cfg["p"])
    elif name == "resize":
        trfm = Resize(height=cfg["height"],
                      width=cfg["width"],
                      p=cfg["p"])
    elif name == "blur":
        trfm = Blur(p=cfg["p"])
    elif name == "color_jitter":
        trfm = ColorJitter(brightness=cfg["brightness"],
                           contrast=cfg["contrast"],
                           saturation=cfg["saturation"],
                           p=cfg["p"])
    elif name == "to_gray":
        trfm = ToGray(p=cfg["p"])

    elif name == "random_shadow":
        trfm = RandomShadow(shadow_roi=[cfg["x_min"], cfg["y_min"], cfg["x_max"], cfg["y_max"]],
                            num_shadows_lower=cfg["num_shadows_lower"],
                            num_shadows_upper=cfg["num_shadows_upper"],
                            shadow_dimension=cfg["shadow_dimension"],
                            always_apply=cfg["always_apply"],
                            p=cfg["p"])
    elif name == "grid_distortion":
        trfm = GridDistortion(p=cfg["p"],
                              # border_mode=cv.BORDER_REPLICATE
                              border_mode=cv2.BORDER_CONSTANT
                              )
    elif name == "elastic_transform":
        trfm = ElasticTransform(p=cfg["p"],
                                # border_mode=cv.BORDER_REPLICATE,
                                border_mode=cv2.BORDER_CONSTANT
                                )
    elif name == "random_rotate_90":
        trfm = RandomRotate90(cfg["p"])
    elif name == "normalize":
        trfm = Normalize(
            p=cfg["p"],
            always_apply=cfg["always_apply"],
            max_pixel_value=255
        )
    elif name == "rotate":
        trfm = Rotate(
            p=cfg["p"],
            always_apply=cfg["always_apply"],
            border_mode=cv2.BORDER_CONSTANT
        )
    elif name == "random_resized_crop":
        trfm = RandomResizedCrop(
            height=cfg["height"],
            width=cfg["width"],
            scale=[cfg["min_scale"], cfg["max_scale"]],
            always_apply=cfg["always_apply"],
            p=cfg["p"]
        )
    elif name == "rgb_shift":
        trfm = RGBShift(
            r_shift_limit=cfg["r_shift_limit"],
            g_shift_limit=cfg["g_shift_limit"],
            b_shift_limit=cfg["b_shift_limit"],
            always_apply=cfg["always_apply"],
            p=cfg["p"]
        )
    else:
        msg = f"Transform {name} is wrong."
        log.critical(msg)
        raise Exception(msg)

    return trfm


def cfg2trfm(cfg):
    """
    Convert config to the augmentation as Composition of transformations.
    """
    trfm_list = [cfg2mini_trfm(mini_trfm_cfg) for mini_trfm_cfg in cfg]
    return A.Compose(trfm_list)
