import torch
import torch.optim
from torch.utils.data import DataLoader
import pandas as pd

import os

from tqdm import tqdm


import logging

from .dataset_tools import MultiImgMaskSet
from .optim_tools import CustomOptimizer, CustomScheduler
from .early_stopper import EarlyStopper
from .metric_tools import cfg2metric_dict
from .loss_tools import cfg2loss

log = logging.getLogger(__name__)


def fit(epochs: int, val_period: int, save_path: str,
        model, loss, metrics: dict,
        optimizer: CustomOptimizer, scheduler: CustomScheduler,
        early_stopper: EarlyStopper,
        train_loader: DataLoader, val_loader: DataLoader):

    train_loss_history = []
    val_loss_history = []
    lr_history = {}
    for i, group in enumerate(optimizer.get_param_groups()):
        lr_history[f"group{i}_lr"] = []

    for epoch in range(epochs):
        model.train()
        epoch_size_counter = 0
        loss_sum = 0
        log.info(f"===== TRAIN (Epoch: {epoch}) =====")
        for dir_names, img_names, img_batch, mask_batch in tqdm(train_loader):
            optimizer.zero_grad()
            predicted_batch = model.inference(img_batch)
            loss_value = loss(predicted_batch, mask_batch)
            loss_sum += float(loss_value.data)
            epoch_size_counter += 1

            loss_value.backward()
            optimizer.step()

        del dir_names, img_names, img_batch, mask_batch, predicted_batch

        log.info(f"Loss: {loss_sum/epoch_size_counter}")

        # on first and last epochs validation happens too
        if epoch % val_period == 0 or epoch == epochs:
            log.info(f"===== Validation (Epoch: {epoch}) =====")

            train_loss_history.append(loss_sum / epoch_size_counter)
            # for every validation save metrics for each photo.
            metric_history = {}
            for metric_name in metrics:
                metric_history[metric_name] = []

            # save model weights
            torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            model.eval()
            loss_sum = 0
            n = 0
            metric_history = {}
            for metric_name in metrics:
                metric_history[metric_name] = []

            dir_name_list = []
            img_name_list = []
            with torch.no_grad():
                for dir_names, img_names, img_batch, mask_batch in tqdm(val_loader):

                    dir_name_list = dir_name_list + list(dir_names)
                    img_name_list = img_name_list + list(img_names)

                    predicted_batch = model.inference(img_batch)
                    loss_value = loss(predicted_batch, mask_batch)
                    loss_sum += float(loss_value.data)
                    n += 1
                    for metric_name, metric in metrics.items():
                        metric_values = metrics[metric_name](predicted_batch[0], mask_batch).cpu().reshape([-1]).tolist()
                        metric_history[metric_name] = metric_history[metric_name] + metric_values

            del dir_names, img_names, img_batch, mask_batch, predicted_batch

            # saving all metric results for each img
            results = metric_history
            results["dir"] = dir_name_list
            results["img"] = img_name_list
            res_df = pd.DataFrame(results)
            res_df.to_excel(os.path.join(save_path, f"metrics{epoch}.xlsx"), index=False)

            val_loss_history.append(loss_sum/n)
            for i, group in enumerate(optimizer.get_param_groups()):
                lr_history[f"group{i}_lr"].append(group["lr"])

            scheduler.step(loss_sum / n)

            log.info(f"Loss: {loss_sum/n}")

            early_stopper(res_df, model)

            history_df = pd.DataFrame({
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
                **lr_history
            })

            history_df.to_excel(os.path.join(save_path, "history.xlsx"), index=False)

            if early_stopper.early_stop:
                break


def cfg2fit(cfg, model, train_dataset: MultiImgMaskSet, val_dataset: MultiImgMaskSet):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg["train_batch_size"],
                              drop_last=True,
                              )

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg["val_batch_size"])

    val_period = cfg["val_period"]
    epochs = cfg["epochs"]

    loss = cfg2loss(cfg["loss"])
    metrics = cfg2metric_dict(cfg["metrics"])

    # check if target_metric is in metrics
    if cfg["early_stopper"]["target_metric"] not in metrics.keys():
        msg = f"Main metric {cfg['target_metric']} not in metrics."
        log.critical(msg)
        raise Exception(msg)

    optimizer = CustomOptimizer(cfg["optimizer"], model.get_params())
    scheduler = CustomScheduler(cfg["scheduler"], optimizer.optimizer)
    early_stopper = EarlyStopper(cfg["early_stopper"])

    fit(epochs=epochs, val_period=val_period, save_path=cfg["save_path"],
        model=model, loss=loss, metrics=metrics,
        optimizer=optimizer, scheduler=scheduler,
        early_stopper=early_stopper,
        train_loader=train_loader, val_loader=val_loader)



