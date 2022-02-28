#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from setuptools import setup, find_packages

setup(name='world-of-bugs-experiments',
      version='0.0.1',
      description='',
      url='',
      author='Benedict Wilkins',
      author_email='benrjw@gmail.com',
      packages=find_packages(),
      install_requires=[
                        #'tml @ git+https://git@github.com/BenedictWilkins/pytorch-module-lib.git',
                        #'gymu @ git+https://git@github.com/BenedictWilkins/gymu.git',
                        'numpy', 
                        'torch', 
                        'wandb',
                        'torchvision', 
                        'webdataset', 
                        'pytorch-lightning',
                        'kaggle',
                        'hydra-core',
                        'scikit-learn',
                        'matplotlib',
                        ],
      zip_safe=False)


