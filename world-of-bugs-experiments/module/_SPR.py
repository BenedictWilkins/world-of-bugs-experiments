#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 03-02-2022 13:21:27

    Modules used in SPR.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from hydra.utils import instantiate

import tml
from tml.discrete import DiscreteLinear

__all__ = ("SPRLightningModule", "P2", "MLP3", "SPR", "DiscreteLinear")

class SPRLightningModule(LightningModule): 
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.criterion = self.cosin_similarity # TODO get from cfg? 
        self.save_hyperparameters()

    def configure_optimizers(self):
        return instantiate(self.cfg.optimiser, self.parameters())

    def training_step(self, batch, i):
        batch_state, batch_action = batch
        loss = sum(self._seq_loss(batch_state, batch_action)).mean()
        self.log("loss", loss.item(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, i):
        batch_state, batch_action = batch
        losses = {f"step-{i}":loss.mean() for i,loss in enumerate(self._seq_loss(batch_state, batch_action))}
        self.log("Validation/loss", losses)

    def test_step(self, batch, i):
        batch_state, batch_action = batch

    def _seq_loss(self, x, a):
        for zi_, zi in self.model._seq_predict(x, a):
            yield self.criterion(zi_, zi)
        
    def cosin_similarity(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1) # [0,4] where 0 is the same, minimise this!


# =================== TORCH MODULES ===================





class P2(nn.Sequential): 
    """
        A simple 2-layer network with optional batch normalisation that may be used for projection or prediction.
    """
    def __init__(self,
                 input_shape : int, 
                 projection_shape : int, 
                 hidden_shape : int = 4096, 
                 track_running_stats : bool = True, 
                 batch_norm : bool = True):
  
        self.input_shape = tml.shape.as_shape(input_shape)
        self.projection_shape = tml.shape.as_shape(projection_shape)
        self.hidden_shape = tml.shape.as_shape(hidden_shape)
        super().__init__(
            nn.Linear(self.input_shape[0], self.hidden_shape[0]),
            nn.BatchNorm1d(self.hidden_shape[0], track_running_stats=track_running_stats) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_shape[0], self.projection_shape[0])
        )

class MLP3(nn.Sequential):
    """
        A simple 3-layer MLP that may be used for encoding.
    """
    def __init__(self, 
                input_shape : tuple, 
                output_shape : tuple, 
                hidden_shape : tuple = (4096,), 
                dropout : float = 0.5):
        self.input_shape = tml.shape.as_shape(input_shape)
        self.output_shape = tml.shape.as_shape(output_shape)
        self.hidden_shape = tml.shape.as_shape(hidden_shape)
        view = tml.View(self.input_shape, -1)
        super().__init__(view, 
            nn.Linear(view.output_shape[0], self.hidden_shape[0]), nn.LeakyReLU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(), 
            nn.Linear(self.hidden_shape[0], self.hidden_shape[0]), nn.LeakyReLU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  
            nn.Linear(self.hidden_shape[0], self.output_shape[0])      
        )

class SPR(nn.Module): 
    
    def __init__(self, 
                encoder : nn.Module, 
                state_shape : tuple,
                action_shape : tuple, 
                transition = None, 
                projector = None, 
                predictor = None, 
                projection_shape : tuple = (256,), 
                hidden_shape : tuple = (1024,)):
        super().__init__()
        self.state_shape = tml.shape.as_shape(state_shape)
        self.action_shape = tml.shape.as_shape(action_shape) 
        self.projection_shape = tml.shape.as_shape(projection_shape)
        self.hidden_shape = tml.shape.as_shape(hidden_shape)
        
        self._encoder = encoder
        
        test_state_input = torch.zeros(2, *state_shape).to(self.device)
        test_action_input = torch.zeros(2, 1).to(self.device)  # dont know if the model accepts one-hot or integer actions... ??
        encoder_output = self.encode(test_state_input)
        self._transition =  tml.discrete.DiscreteLinear(encoder_output.shape[1:], self.action_shape).to(self.device) if transition is None else transition
        transition_output = self.transition(encoder_output, test_action_input)
        self._projector = P2(transition_output.shape[1:], self.projection_shape, self.hidden_shape).to(self.device) if projector is None else projector
        projector_output = self.project(transition_output)
        self.projection_shape = tml.shape.as_shape(projector_output.shape[1:])
        self._predictor = P2(self.projection_shape, self.projection_shape, self.hidden_shape).to(self.device) if predictor is None else predictor
    
    @property
    def device(self):
        return next(self._encoder.parameters()).device

    def forward(self, state, action):
        target, prediction = torch.cat([x for x in zip(*self._seq_predict(state, action))])
        return target, prediction
                
    def _seq_predict(self, s, a):
        for zi, zi_ in self._seq_project(s, a):
            yield zi, self.predict(zi_)
    
    def _seq_project(self, s, a):
        for zi, zi_ in self._seq_transition(s, a):
            yield self.target_project(zi), self.project(zi_)
            
    def _seq_transition(self, s, a):
        z = self.encode(s).transpose(0,1)  #  [sequence_size, batch_size, ...]
        a = a.transpose(0,1)               # [sequence_size, batch_size, ...]
        zi_ = z[0]
        z, a = z[1:], a[:-1]
        for i in range(z.shape[0]):
            zi, ai = z[i], a[i]
            zi_ = self.transition(zi_, ai)
            yield zi, zi_ # target, prediction
    
    def encode(self, x):
        z = self._encoder(x.view(-1, *self.state_shape))
        return z.view(*x.shape[:-len(self.state_shape)], *z.shape[1:])
    
    def transition(self, x, a):
        return self._transition(x, a)
        
    def predict(self, x):
        return self._predictor(x)
    
    def project(self, x):
        return self._projector(x)
    
    def target_project(self, x):
        return self._projector(x).detach() # TODO EMA option? 




