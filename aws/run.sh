#!/bin/sh

# =====================
# Use the Deep Learning Base AMI (ubuntu)
# g3.4xlarge
# dont forget to add storage! (atleast 100gb, 200gb to be safe)
# =====================

DIRECTORY="/home/ubuntu/.aws"
# sync install files
rsync -rv --inplace ./ ubuntu@$1:$DIRECTORY

# wait for dpkg to become avaliable...
#ssh ubuntu@$1 "source $DIRECTORY/waitfordpkg.sh && exit"
# update and install packages
#ssh ubuntu@$1 "source $DIRECTORY/install_dpkg.sh $DIRECTORY && exit"

# install anaconda, restarting the connection creates another shell, which is important for anaconda setup...
# this is only needed if ec2 instance doesnt already have anaconda installed!
ssh ubuntu@$1 "source $DIRECTORY/install_anaconda.sh && exit"

# setup kaggle & wandb 
rsync -rv --inplace ~/.wandb/ ubuntu@$1:/home/ubuntu/.wandb/
rsync -rv --inplace ~/.kaggle/ ubuntu@$1:/home/ubuntu/.kaggle/

# install python packages
ssh ubuntu@$1 "source $DIRECTORY/install_python.sh  && exit"

# do your thing!
ssh ubuntu@$1 

