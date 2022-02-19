#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"



import pathlib
import glob
import more_itertools
import itertools

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
from pytorch_lightning import LightningDataModule

from kaggle.api.kaggle_api_extended import KaggleApi

import gymu

__all__ = ("WOBDataModule",)

WORLD_OF_BUGS_DATASET_KAGGLE = "benedictwilkinsai/world-of-bugs"

DATASET_VERSION = -1 # TODO include this! always use latest version...
TRAINING_SET = "NORMAL-TRAIN/*.tar.gz"
VALIDATION_SET = "NORMAL-TRAIN-SMALL/*-0000.tar.gz"
TEST_SET = "TEST/**/*.tar.gz"

class WOBDataModule(LightningDataModule):

    def __init__(self, 
                    path,
                    force=False, 
                    batch_size=256, 
                    train_mode=['state', 'action', 'next_state'],
                    validation_mode=None,
                    test_mode=None,
                    shuffle_buffer_size = 10000,
                    initial_buffer_size = 10000,
                    num_workers = 12,
                    prefetch_factor = 2,
                    window_size = 1,
                    train_files = TRAINING_SET,
                    validation_files = VALIDATION_SET,
                    test_files = TEST_SET,
                    ):
        super().__init__()
        self.path = pathlib.Path(pathlib.Path(path).resolve(), WORLD_OF_BUGS_DATASET_KAGGLE)
        self.force = force
        self.batch_size = batch_size

        self.train_mode = gymu.mode.mode(train_mode) # cast training datasets to this mode
        self.validation_mode = gymu.mode.mode(validation_mode) if validation_mode is not None else self.train_mode
        self.test_mode = gymu.mode.mode(test_mode) if test_mode is not None else self.train_mode

        self.shuffle_buffer_size = shuffle_buffer_size
        self.initial_buffer_size = initial_buffer_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.window_size = window_size # if in time series mode... this should be the same for train,test,validation

        self.train_files, self.validation_files, self.test_files = train_files, validation_files, test_files

    def prepare_data(self):
        if self.force or not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            api = KaggleApi()
            api.authenticate() # requires ~/.kaggle/kaggle.json file
            api.dataset_download_files(WORLD_OF_BUGS_DATASET_KAGGLE, quiet=False, unzip=True, force=self.force, path=str(self.path))

        # resolve files...
        def resolve_files(files): 
            if isinstance(files, str): # a suitable glob pattern
                _path = pathlib.Path(self.path, files).resolve()
                files = list(sorted(glob.glob(str(_path), recursive=True)))
            else:
                raise TypeError("Invalid files specified: {files}, please provide a valid glob pattern.")
            return files
        
        self.train_files = resolve_files(self.train_files)
        self.validation_files = resolve_files(self.validation_files)
        self.test_files = resolve_files(self.test_files)

    def train_dataloader(self):
        dataset = gymu.data.dataset(self.train_files, shardshuffle=True)
        dataset = dataset.then(gymu.data.Composable.decode(keep_meta=True))
        dataset = dataset.then(gymu.data.Composable.keep(self.train_mode.keys()))
        if self.window_size > 1:
            dataset.then(gymu.data.Composable.window(self.window_size))
        dataset = dataset.then(gymu.data.Composable.mode(self.train_mode))
        dataset = dataset.shuffle(size=self.shuffle_buffer_size, initial=self.initial_buffer_size)
        dataset = dataset.batched(self.batch_size)
        return DataLoader(dataset, 
                            batch_size=None, 
                            num_workers=self.num_workers, 
                            prefetch_factor=self.prefetch_factor,
                            persistent_workers=True, 
                            pin_memory=True)

    def val_dataloader(self):
        dataset = gymu.data.dataset(self.validation_files)
        dataset = dataset.then(gymu.data.Composable.decode(keep_meta=False))
        dataset = dataset.then(gymu.data.Composable.keep(self.validation_mode.keys()))
        if self.window_size > 1:
            dataset.then(gymu.data.Composable.window(self.window_size))
        dataset = dataset.then(gymu.data.Composable.mode(self.validation_mode))
        dataset = dataset.batched(self.batch_size)
        return DataLoader(dataset, 
                        batch_size=None, 
                        num_workers=self.num_workers, 
                        prefetch_factor=self.prefetch_factor,
                        persistent_workers=True, 
                        pin_memory=True)

    def test_dataloader(self):

        #pprint(self.test_files)
        ## group by bug type
        #files = itertools.groupby([p for p in sorted(self.test_files)], key=lambda k: str(pathlib.Path(k).parent.stem))
        #files = {k:list(v) for k,v in files}
        
        def url_to_info(x):
            x['info'] = [*x['info'], np.array([x["__url__"].split("/")[-2]])]
            return x
        def collate(x):
            state = torch.stack([torch.from_numpy(z.state) for z in x])
            info = [torch.stack([torch.from_numpy(z.info[i]) for z in x]) for i in range(len(x[0].info)-1)]
            return (state, *info, x[0].info[-1][0])

        dataset = gymu.data.dataset(self.test_files)
        dataset = dataset.then(gymu.data.Composable.decode(keep_meta=True))
        dataset = dataset.map(url_to_info)
        dataset = dataset.then(gymu.data.Composable.keep(self.test_mode.keys()))

        if self.window_size > 1:
            dataset.then(gymu.data.Composable.window(self.window_size))
        dataset = dataset.then(gymu.data.Composable.mode(self.test_mode))
        dataset = dataset.batched(self.batch_size, collation_fn=collate)

        return DataLoader(dataset, 
                        batch_size=None, 
                        num_workers=self.num_workers, 
                        prefetch_factor=self.prefetch_factor,
                        persistent_workers=True, 
                        pin_memory=True)

if __name__ == "__main__":
    from pprint import pprint
    dm = WOBDataModule(path="./dataset", force=False, batch_size=2, num_workers=2, test_mode=['state', 'info'])
    dm.prepare_data()
    loader = dm.test_dataloader()

    for s, _, _, i in loader:
        print(i)

        break




    

