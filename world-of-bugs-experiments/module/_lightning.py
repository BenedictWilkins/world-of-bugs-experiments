#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"


from types import SimpleNamespace
import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate

from tml.utils import as_shape

from ._module import AE

__all__ = ("AELightningModule",)

class AELightningModule(pl.LightningModule):
    
    def __init__(self, model, criterion, optimiser):
        super().__init__()
        self.cfg = SimpleNamespace(model_cfg=model, criterion_cfg=criterion, optimiser_config=optimiser)
        self.model = instantiate(model)
        self.criterion = instantiate(criterion)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return instantiate(self.cfg.optimiser_cfg, self.parameters())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_i):
        pass 