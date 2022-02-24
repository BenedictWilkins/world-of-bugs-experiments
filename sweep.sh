
python -m world-of-bugs-experiments '++lightning.criterion._target_=world-of-bugs-experiments.module.MSESSIM' '++lightning.criterion.window_size=7' '++lightning.optimiser.lr=0.0005' '++lightning.score._target_=torch.nn.MSELoss' '++trainer.max_epochs=5' '++lightning.model._target_=world-of-bugs-experiments.module.AEConv'

#python -m world-of-bugs-experiments '++lightning.criterion._target_=world-of-bugs-experiments.module.SSIM' '++lightning.criterion.window_size=7' '++lightning.optimiser.lr=0.0005' '++trainer.max_epochs=20' '++lightning.model._target_=world-of-bugs-experiments.module.AEAlex'

#python -m world-of-bugs-experiments '++lightning.criterion._target_=world-of-bugs-experiments.module.SSIM' '++lightning.criterion.window_size=7' '++lightning.optimiser.lr=0.0005' '++trainer.max_epochs=20' '++lightning.model._target_=world-of-bugs-experiments.module.AEGDN'
