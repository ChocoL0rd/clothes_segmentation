import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import os


__all__ = [
    "cfg2model"
]

# model_name: model
name2model_class = {
    # model discription like in segmentation library
    "unet": smp.Unet,
    "manet": smp.MAnet,
    "unetpluplus": smp.UnetPlusPlus,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3
}


def cfg2model(cfg):
    """ Returns model according to it config """
    model = name2model_class[cfg["name"]](
            encoder_name=cfg["encoder_name"],
            encoder_weights=cfg["encoder_weights"],
            in_channels=3,
            classes=1,
            activation="sigmoid",
    ).to(cfg["device"])
    
    
    if cfg["load_pretrained"]:
        model.load_state_dict(torch.load(os.path.join(cfg["pretrained_path"], "model.pt")))

    return model



