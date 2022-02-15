#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"


from bz2 import compress
from typing import Callable, List
from tqdm.auto import tqdm

import itertools
import h5py
import pathlib

import numpy as np

import torch

from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
from pytorch_lightning import LightningDataModule

import os
import json
import glob

from kaggle.api.kaggle_api_extended import KaggleApi

import gymu


WORLD_OF_BUGS_DATASET_KAGGLE = "benedictwilkinsai/world-of-bugs"

DATASET_TRAINING_SET_FILE = "NORMAL-300k.hdf5"
DATASET_VALIDATION_SET_FILE = "NORMAL-50k.hdf5"
DATASET_TEST_SET_DIR = "TEST"


class WOBDataModule(LightningDataModule):

    def __init__(self, path, force=False, batch_size=256, num_episodes=5):
        super().__init__()
        self.path = pathlib.Path(path).resolve()
        self.force = force
        self.batch_size = batch_size
        self.num_episodes = num_episodes

        self._train_files = []
        self._test_files = []
    
    def _get_episodes(self, path, n, keys=None, lazy=True):
        # make data keys gymu compatible...
        #gymu_keys = [k.replace("observation", "state") for k in keys]
        #gymu_keys = [k.replace("bugmask", "info") for k in gymu_keys]
        #gymu_mode = gymu.mode.mode(gymu_keys) # data mode... now can use gymu datasets!
        #gymu_mode_map = {k:v for k,v in zip(keys,gymu_keys)}

        def _get_data(group):
            data = {k:group[k] for k in group.keys()}
            print(data.keys())
       


        with h5py.File(path, "r") as f:
            ep_iterator = itertools.islice(sorted(f.keys()), n)
            episodes = [_get_data(f[ep_name]) for ep_name in ep_iterator]
            return episodes
            
    def prepare_data(self):
        api = KaggleApi()
        api.authenticate() # requires ~/.kaggle/kaggle.json file
        api.dataset_download_files(WORLD_OF_BUGS_DATASET_KAGGLE, quiet=False, unzip=True, force=self.force, path=str(self.path))
        # decompress dataset...? 

        # read files in path
        path = str(self.path) + "/**/*.hdf5" 
        print(f"DECOMPRESSING FILES...")
        for file in glob.glob(path, recursive=True):
            if self.force or pathlib.Path(file).exists():
                episodes, f = gymu.data.read_episodes(file, lazy=True, show_progress=False)
                gymu.data.write_episodes(file.replace("-compressed", ""), episodes, compression=None, show_progress=True)
                f.close()

    def train_dataloader(self):
        episodes = self._get_episodes(pathlib.Path(self.path, DATASET_TRAINING_SET_FILE), self.num_episodes, keys=self.dataset_cls.__keys__)
        dataset = gymu.data.EpisodeDataset(episodes)
        loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)
        return loader

    def val_dataloader(self):
        pass 

    def test_dataloader(self):
       pass 

if __name__ == "__main__":
    dm = WOBDataModule(path="./dataset", force=False)
    dm.prepare_data()
    #dm.train_dataloader()




    

