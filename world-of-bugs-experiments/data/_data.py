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

WORLD_OF_BUGS_DATASET_KAGGLE = ["benedictwilkinsai/world-of-bugs-normal", "benedictwilkinsai/world-of-bugs-test"]
WORLD_OF_BUGS_DATASET_PATH = "benedictwilkinsai/world-of-bugs/dry"

DATASET_VERSION = -1 # TODO include this! always use latest version...
TRAINING_SET = "NORMAL-TRAIN/*/*.tar"
VALIDATION_SET = "NORMAL-TRAIN-SMALL/*/*-0000.tar"
TEST_SET = "TEST/**/*.tar"

class WOBDataModule(LightningDataModule):

    def __init__(self, 
                    path=".",
                    force=False, 
                    batch_size=256, 
                    in_memory = False,
                    train_mode=['state', 'action'],
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
        self.path = pathlib.Path(pathlib.Path(path).resolve(), WORLD_OF_BUGS_DATASET_PATH)
        print(self.path)
        self.force = force
        self.batch_size = batch_size
        self.in_memory = in_memory

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
            for url in WORLD_OF_BUGS_DATASET_KAGGLE:
                api.dataset_download_files(url, quiet=False, unzip=True, force=self.force, path=str(self.path))
                print("Done.")
        # resolve files...
        def resolve_files(files, label=""): 
            if isinstance(files, str): # a suitable glob pattern
                _path = pathlib.Path(self.path, files).resolve()
                files = list(sorted(glob.glob(str(_path), recursive=True)))
            elif files is None:
                files = []
            else:
                raise TypeError("Invalid files specified: {files}, please provide a valid glob pattern.")
            print(f"Found {len(files)} {label} files.")
            return files
        
        self.train_files = resolve_files(self.train_files, 'training')
        self.validation_files = resolve_files(self.validation_files, 'validation')
        self.test_files = resolve_files(self.test_files, 'test')

        self.train_dataset = self._intialise_train_dataset()
        self.validation_dataset = self._initialise_validation_dataset()
        self.test_dataset = self._initialise_test_dataset()

    def _initialise_base_dataset(self, files, mode, in_memory, shuffle):
        if len(files) > 0:
            dataset = gymu.data.dataset(files, shardshuffle=True)
            dataset = dataset.gymu.decode(keep_meta=False)
            dataset = dataset.gymu.keep(mode.keys())
            dataset = dataset.gymu.window(self.window_size) if self.window_size > 1 else dataset
            dataset = dataset.gymu.mode(mode)
            dataset = dataset.shuffle(self.shuffle_buffer_size, initial=self.initial_buffer_size) if shuffle and not in_memory else dataset
            dataset = dataset.gymu.to_tensor_dataset(num_workers=self.num_workers, show_progress=True) if in_memory else dataset.batched(self.batch_size)
            return dataset
        return None # not using this dataset? TODO give a warning?

    def _intialise_train_dataset(self):
        return self._initialise_base_dataset(self.train_files, self.train_mode, self.in_memory, True)

    def _initialise_validation_dataset(self):
        return self._initialise_base_dataset(self.validation_files, self.validation_mode, False, False)
        
    def _initialise_test_dataset(self):
        def url_to_info(x):
            x['info'] = [*x['info'], np.array([x["__url__"].split("/")[-3]])]
            return x
        def collate(x):
            state = torch.stack([torch.from_numpy(z.state) for z in x])
            info = [torch.stack([torch.from_numpy(z.info[i]) for z in x]) for i in range(len(x[0].info)-1)]
            return (state, *info, x[0].info[-1][0])

        dataset = gymu.data.dataset(self.test_files)
        dataset = dataset.gymu.decode(keep_meta=True)
        dataset = dataset.map(url_to_info)
        dataset = dataset.gymu.keep(self.test_mode.keys())
        dataset = dataset.gymu.window(self.window_size) if self.window_size > 1 else dataset
        dataset = dataset.gymu.mode(self.test_mode)
        dataset = dataset.batched(self.batch_size, collation_fn=collate)
        return dataset

    def train_dataloader(self):
        if self.in_memory:
            return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, pin_memory=True)
        else:
            return DataLoader(self.train_dataset, shuffle=False, batch_size=None, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=None, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

if __name__ == "__main__":
    from pprint import pprint
    dm = WOBDataModule(path="./dataset", force=False, batch_size=2, num_workers=2, test_mode=['state', 'info'])
    dm.prepare_data()





    

