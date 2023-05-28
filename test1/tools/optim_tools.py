from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import Adam, SGD

import logging

log = logging.getLogger(__name__)


class CustomOptimizer:
    def __init__(self, cfg, model_parts):
        param_list = []
        if len(model_parts) != len(cfg["lrs"]) or len(model_parts) != len(cfg["weight_decays"]):
            msg = f"Number of model_parts {len(model_parts)}, learning rates {len(cfg['lrs'])} " \
                  f"and weight_decays {cfg['weight_decays']} should be the same."
            log.critical(msg)
            raise Exception(msg)

        for i in range(len(cfg["lrs"])):
            param_list.append(
                {
                    "params": model_parts[i],
                    "lr": cfg["lrs"][i],
                    "weight_decay": cfg["weight_decays"][i]
                }
            )

        if cfg["name"] == "sgd":
            self.optimizer = SGD(param_list, 1)
        elif cfg["name"] == "adam":
            self.optimizer = Adam(param_list)
        else:
            msg = f"Optimizer {cfg['name']} is not supported"
            log.critical(msg)
            raise ValueError(msg)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def get_param_groups(self):
        return self.optimizer.param_groups


class CustomScheduler:
    def __init__(self, cfg, optimizer):
        self.scheduler_name = cfg["name"]

        if self.scheduler_name == 'step_lr':
            self.scheduler = StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["factor"])
        elif self.scheduler_name == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(optimizer, factor=cfg["factor"])
        else:
            msg = f"Scheduler {cfg['name']} not supported"
            log.critical(msg)
            raise ValueError(msg)

    def step(self, val_loss):
        if self.scheduler_name == "reduce_on_plateau":
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()


