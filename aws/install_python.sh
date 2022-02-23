#!/bin/sh

# weird bugs here...? need to restart the terminal for it to work properly? ehh
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc

conda create -n PhD -y python=3.7
conda activate PhD

conda install pytorch cudatoolkit=10.2 -c pytorch -y

cd ~
git clone https://github.com/BenedictWilkins/world-of-bugs-experiments.git
pip install -e world-of-bugs-experiments

# setup wandb key
cd ~
WANDBKEY=$(<.wandb/wandb.txt)
wandb login $WANDBKEY

