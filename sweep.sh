python -m world-of-bugs-experiments '++lightning.criterion._target_=torch.nn.BCEWithLogitsLoss' '++lightning.score._target_=torch.nn.MSELoss'
python -m world-of-bugs-experiments '++lightning.criterion._target_=torch.nn.MSELoss'
python -m world-of-bugs-experiments '++lightning.criterion._target_=world-of-bugs-experiments.module.SSIM' '++lightning.criterion.window_size=7'
