import hydra
import hydra.core.hydra_config
from omegaconf import OmegaConf

import torch
import numpy as np

import os
import logging

# import my tools
from tools.model_tools import cfg2model
from tools.test_tools import cfg2test
from tools.dataset_tools import cfg2datasets

import random

# fix random seeds to make results reproducible.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_path = hydra_cfg.runtime.output_dir
    OmegaConf.resolve(cfg)

    # creating model
    model = cfg2model(cfg["model_cfg"])

    # creating datasets
    datasets = cfg2datasets(cfg["dataset_cfg"])

    # here implemented illustration of images and result checking of interpolated
    cfg2test_with_png(cfg["test_cfg"], model, datasets["test"])
    cfg2test_with_png(cfg["test_cfg"], model, datasets["train"])

if __name__ == "__main__":
    my_app()


