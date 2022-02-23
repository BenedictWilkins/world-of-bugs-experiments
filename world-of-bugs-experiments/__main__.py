#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import hydra
import omegaconf
from hydra.utils import instantiate

import wandb

from pytorch_lightning import Trainer


@hydra.main(config_name="config_AE.yaml", config_path="./configuration")
def main(cfg) -> None:

    
    datamodule = instantiate(cfg.data)
    
    model = instantiate(cfg.lightning, _recursive_=False)

    wandb.init(
        project=cfg.trainer.logger.project, 
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    trainer = instantiate(cfg.trainer)
   
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    wandb.finish() # prevents hanging at the end...?

if __name__ == "__main__":
    main()



"""

TODO

1. Run experiments on AWS
4. Update paper conclusions and TODOs get final version !!
2. Install unity 
3. new WOB build
4. generate dataset
5. update documentation for WOB in code and on github.
6. update kaggle dataset on github

WHEN PAPER IS DONE!! put on archive and update all referenes to it.
SUBMIT TO COG 2022 ON OR BEFORE FRIDAY!!

"""