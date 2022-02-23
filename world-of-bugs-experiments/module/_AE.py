#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from re import L
from types import SimpleNamespace
from typing import DefaultDict

import torch
import torch.nn as nn

from tml import ResBlock2D, View
from tml.utils import as_shape

import torchvision

import pytorch_lightning as pl
from hydra.utils import instantiate

from pprint import pprint

__all__ = ("AE", "AELightningModule")


class AELightningModule(pl.LightningModule):
    
    def __init__(self, model, criterion, score, optimiser, metrics=[]):
        super().__init__()
        self.model = instantiate(model)
        self.criterion = instantiate(criterion)
        self.score = instantiate(score)
        self.metrics = [instantiate(m) for m in metrics]
        self.optimiser =  instantiate(optimiser, self.parameters())
        self.reconstruction_as_image = reconstruction_as_image(self)

    def configure_optimizers(self):
        return self.optimiser

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_i):
        state, = batch
        loss = self.criterion(self(state), state)
        self.log("train/loss", loss.item())
        return loss
        
    def validation_step(self, batch, batch_i):
        state, = batch
        reconstruction = self(state)
        self.log("validation/loss", self.criterion(reconstruction, state).item())
        if batch_i == 0: # log images on the first batch
            show_n = 8
            img_reconstruction, img_state = self.reconstruction_as_image(reconstruction[:show_n]), state[:show_n]
            img = torchvision.utils.make_grid([*img_reconstruction, *img_state], nrow=show_n)
            img = torch.clip(img, 0, 1)
            self.logger.log_image("validation/reconstruction", [img])

    def test_step(self, batch, batch_i):
        state, _, bugmask, bug_type = batch
        label = bugmask.sum(1, keepdims=True) > 0.
        label = label.view(label.shape[0], -1).sum(-1).cpu()
        score = self.score(self(state), state)
        score = score.view(score.shape[0], -1).sum(-1).cpu()
        return (bug_type, (score, label))

    def test_epoch_end(self, outputs):
        scores = DefaultDict(list)
        labels = DefaultDict(list)
        for bug_type, (score, label) in outputs:
            scores[bug_type].append(score)
            labels[bug_type].append(label)
        scores = {k:torch.cat(v) for k,v in scores.items()}
        labels = {k:torch.cat(v) for k,v in labels.items()}

        assert set(scores.keys()) == set(labels.keys())
        for metric in self.metrics:# for each score compute the metric
            for k in scores.keys(): 
                x, y = scores[k], labels[k]
                assert x.shape == y.shape
                metric(self.logger.experiment, x, y, title = f"{k} {metric.abreviation}")
                

            

def reconstruction_as_image(model): # ensures that a reconstruction is done properly when testing purposes
    if "logit" in str(model.criterion).lower():
        def _recon(x):
            return torch.sigmoid(x)
        return _recon
    else:
        def _recon(x):
            return x
        return _recon

# =================== TORCH MODULES ===================

class Sequential(nn.Sequential):
    
    def __init__(self, input_shape, output_shape, layers):
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        super().__init__(*layers)       


class AE(nn.Module):
    
    def __init__(self, input_shape, latent_shape=1024, channels=16, dropout=0.5, output_layer=None):
        super().__init__()
        assert input_shape == (3,84,84)
        self.input_shape = as_shape(input_shape)
        self.latent_shape = as_shape(latent_shape)
        self.output_shape = self.input_shape
        self.channels = channels
        self.dropout = dropout
        self.output_layer = output_layer if output_layer is not None else nn.Identity()
        self.encoder = self._get_encoder(channels)
        self.decoder = self._get_decoder(channels)
        
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    def _get_encoder(self, channels):
        layers = [
            nn.Conv2d(self.input_shape[0], channels, (6,6), 2), nn.LeakyReLU(),
            ResBlock2D(channels, channels), nn.LeakyReLU(), 
            nn.Conv2d(channels, 2*channels, (3,3), 1), nn.LeakyReLU(),
            nn.Conv2d(2*channels, 2*channels, (3,3), 1), nn.LeakyReLU(),
            nn.Conv2d(2*channels, 2*channels, (2,2), 2), nn.LeakyReLU(), # downsample
            ResBlock2D(2*channels, 2*channels), nn.LeakyReLU(), 
            nn.Conv2d(2*channels, 4*channels, (3,3), 1), nn.LeakyReLU(),
            nn.Conv2d(4*channels, 4*channels, (2,2), 2), nn.LeakyReLU(), # downsample
            nn.Conv2d(4*channels, 4*channels, (3,3), 1), nn.LeakyReLU(),
        ]
        view = View((4*channels,6,6),-1)
        layers.append(view)
        layers.extend([nn.Linear(view.output_shape[0], self.latent_shape[0]), nn.LeakyReLU(),
                        nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
                        nn.Linear(self.latent_shape[0], self.latent_shape[0])])
        return Sequential(self.input_shape, self.latent_shape, layers)

    def _get_decoder(self, channels):
        view = View(-1,(4*channels,6,6))
        layers = [
            nn.LeakyReLU(),
            nn.Linear(self.latent_shape[0], self.latent_shape[0]), nn.LeakyReLU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.latent_shape[0], view.input_shape[0]), nn.LeakyReLU(),
            view,
            nn.ConvTranspose2d(4*channels, 4*channels, (3,3), 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(4*channels, 4*channels, (2,2), 2), nn.LeakyReLU(), 
            nn.ConvTranspose2d(4*channels, 2*channels, (3,3), 1), nn.LeakyReLU(),
            ResBlock2D(2*channels, 2*channels), nn.LeakyReLU(), 
            nn.ConvTranspose2d(2*channels, 2*channels, (2,2), 2), nn.LeakyReLU(), 
            nn.ConvTranspose2d(2*channels, 2*channels, (3,3), 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(2*channels, channels, (3,3), 1), nn.LeakyReLU(),
            ResBlock2D(channels, channels), nn.LeakyReLU(),
            nn.ConvTranspose2d(channels, self.input_shape[0], (6,6), 2),
            self.output_layer,
        ]
        return Sequential(self.latent_shape, self.input_shape, layers) 