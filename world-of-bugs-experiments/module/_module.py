#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch
import torch.nn as nn

from tml import ResBlock2D, View
from tml.utils import as_shape

__all__ = ("AE",)

class Sequential(nn.Sequential):
    
    def __init__(self, input_shape, output_shape, layers):
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        super().__init__(*layers)       

class AE(nn.Module):
    
    def __init__(self, input_shape, latent_shape=1024, channels=16):
        super().__init__()
        assert input_shape == (3,84,84)
        self.input_shape = as_shape(input_shape)
        self.latent_shape = as_shape(latent_shape)
        self.output_shape = self.input_shape
        self.channels = channels
        
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
        layers.append(nn.Linear(view.output_shape[0], self.latent_shape[0]))
        return Sequential(self.input_shape, self.latent_shape, layers)

    def _get_decoder(self, channels):
        view = View(-1,(4*channels,6,6))
        layers = [
            nn.LeakyReLU(),
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
            nn.ConvTranspose2d(channels, self.input_shape[0], (6,6), 2)
        ]
        return Sequential(self.latent_shape, self.input_shape, layers) 

if __name__ == "__main__":
    import torchsummary
    model = AE((3,84,84))
    torchsummary.summary(model, (3,84,84), device="cpu")