import torch
from torch import nn


class Example(nn.Module):
    def __init__(self, cfg):
        super(Example, self).__init__()
        pass

    def forward(self, x):
        pass

    def inference(self, x):
        """ Activated output of model """
        pass

    def get_params(self):
        """ Return parts of model to apply different optimizers to them """
        pass