import torch
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import pandas as pd

from tqdm import tqdm

import os
import logging

from .metric_tools import cfg2metric_dict
from .dataset_tools import MultiImgMaskSet

__all__ = [
    "cfg2test"
]


log = logging.getLogger(__name__)

tensor2pil = ToPILImage()


def cfg2test(cfg, model, dataset: MultiImgMaskSet):
    dataset.set_aug_flag(False)
    dataset.set_return_original(True)

    data_save_path = os.path.join(cfg["save_path"], dataset.log_name)
    os.mkdir(data_save_path)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg["batch_size"],
                            drop_last=False,
                            shuffle=False)

    metrics = cfg2metric_dict(cfg["metrics"])
    metric_history = {}
    for metric_name in metrics:
        metric_history[metric_name] = []

    dir_name_list = []
    img_name_list = []
    model.eval()

    print(f"========== TEST {dataset.log_name}==========")
    with torch.no_grad():
        for dir_name, img_name, img, mask, original_img, original_mask in tqdm(dataloader):
            predicted = model.forward(img)

            dir_name_list = dir_name_list + list(dir_name)
            img_name_list = img_name_list + list(img_name)
            
            interpolated_predict = torch.nn.functional.interpolate(predicted, original_img.shape[-2:])
            interpolated_mask = torch.nn.functional.interpolate(mask, original_img.shape[-2:])
            
            # saving metric history
            for metric_name, metric in metrics.items():
                metric_values = metrics[metric_name](interpolated_predict, interpolated_mask).cpu().reshape([-1]).tolist()                
                metric_history[metric_name] = metric_history[metric_name] + metric_values
            


    # saving all metric results for each img
    results = metric_history
    results["dir"] = dir_name_list
    results["img"] = img_name_list
    res_df = pd.DataFrame(results)
    res_df.to_excel(os.path.join(data_save_path, "full_results.xlsx"), index=False)
    res_df.describe().to_excel(os.path.join(data_save_path, "descr_results.xlsx"))

