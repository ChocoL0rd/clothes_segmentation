import numpy as np
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd

from tqdm import tqdm

import os
import logging

from .metric_tools import cfg2metric_dict
from .dataset_tools import MultiImgMaskSet

__all__ = [
    "cfg2test",
    "cfg2test_with_png"
]


log = logging.getLogger(__name__)

tensor2pil = ToPILImage()


def cfg2test(cfg, model, dataset: MultiImgMaskSet):
    dataset.set_aug_flag(False)
    dataset.set_return_original(True)
    
    data_save_path = os.path.join(cfg["save_path"], dataset.log_name)
    os.mkdir(data_save_path)

    illustrate_path = os.path.join(cfg["save_path"], f"{dataset.log_name}_segmented")
    os.mkdir(illustrate_path)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
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
            predicted = model.inference(img)[0]

            dir_name_list = dir_name_list + list(dir_name)
            img_name_list = img_name_list + list(img_name)
            
            interpolated_predict = torch.nn.functional.interpolate(predicted, original_img.shape[-2:])
            interpolated_mask = torch.nn.functional.interpolate(mask, original_img.shape[-2:])
            
            # illustrating result as a jpg
            masked = original_img * interpolated_predict
            for i, masked_tensor in enumerate(masked):
                tmp_dir_path = os.path.join(illustrate_path, dir_name[i])
                if not os.path.exists(tmp_dir_path):
                    os.mkdir(tmp_dir_path)
            
                masked_pil = tensor2pil(masked_tensor)
                masked_pil.save(os.path.join(tmp_dir_path, img_name[i] + ".jpg"))
            
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



def cfg2test_with_png(cfg, model, dataset: MultiImgMaskSet):
    dataset.set_aug_flag(False)
    dataset.set_return_original(True)
    
    data_save_path = os.path.join(cfg["save_path"], dataset.log_name)
    os.mkdir(data_save_path)

    illustrate_path = os.path.join(cfg["save_path"], f"{dataset.log_name}_segmented")
    os.mkdir(illustrate_path)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
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
            predicted = model.inference(img)[0]

            dir_name_list = dir_name_list + list(dir_name)
            img_name_list = img_name_list + list(img_name)
            
            interpolated_predict = torch.nn.functional.interpolate(predicted, original_img.shape[-2:])
            interpolated_mask = torch.nn.functional.interpolate(mask, original_img.shape[-2:])
            
            # saving png files
            for i in range(img.shape[0]):
                tmp_dir_path = os.path.join(illustrate_path, dir_name[i])
                if not os.path.exists(tmp_dir_path):
                    os.mkdir(tmp_dir_path)
                    
                img_np = original_img[i].cpu().numpy()
                mask_np = interpolated_predict[i].cpu().numpy()
                arr_with_alpha = np.concatenate((img_np, mask_np), axis=0)
                
                image_with_alpha = Image.fromarray(np.uint8(arr_with_alpha.transpose((1, 2, 0)) * 255))
                image_with_alpha.save(os.path.join(tmp_dir_path, img_name[i] + ".png"))
            
                # masked_pil = tensor2pil(masked_tensor)
                # masked_pil.save(os.path.join(tmp_dir_path, img_name[i] + ".jpg"))
            
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



