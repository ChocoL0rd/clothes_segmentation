import logging
import torch
import os

__all__ = [
    "EarlyStopper"
]

log = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, cfg):
        self.patience = cfg["patience"]
        self.delta = cfg["delta"]
        self.metric_name = cfg["target_metric"]

        self.counter = 0
        self.best_metric = None
        self.early_stop = False

        if cfg["statistic"] == "mean":
            self.stat = lambda x: x.mean()
        elif cfg["statistic"] == "max":
            self.stat = lambda x: x.max()

    def __call__(self, metrics, model):
        metric_values = metrics[self.metric_name]
        val_metric = self.stat(metric_values)

        if self.best_metric is None or val_metric > self.best_metric + self.delta:
            # if new loss better than best
            log.info(f"Validation metric changed from {self.best_metric} to {val_metric}")
            self.best_metric = val_metric
            self.counter = 0
        else:
            # if not better
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                log.info("Early stopping.")
