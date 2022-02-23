#!/bin/sh
DIRECTORY="/home/ubuntu/.aws"
# sync install files
rsync -rv --inplace ./ ubuntu@$1:$DIRECTORY

# wait for dpkg to become avaliable...
ssh ubuntu@$1 "source $DIRECTORY/waitfordpkg.sh && exit"
# update and install packages
ssh ubuntu@$1 "source $DIRECTORY/install_dpkg.sh $DIRECTORY && exit"

# install anaconda, restarting the connection creates another shell, which is important for anaconda setup...
ssh ubuntu@$1 "source $DIRECTORY/install_conda.sh && exit"

# install python packages
ssh ubuntu@$1 "source $DIRECTORY/install_python.sh  && exit"




# setup kaggle & wandb 
rsync -rv --inplace ~/.wandb/ ubuntu@$1:/home/ubuntu/.wandb/
rsync -rv --inplace ~/.kaggle/ ubuntu@$1:/home/ubuntu/.kaggle/

ssh ubuntu@1 "source $DIRECTORY/wandb.sh && exit"





# TODO other commands