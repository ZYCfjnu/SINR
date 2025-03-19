import models.local_model as model
import models.data.voxelized_data as voxelized_data
import models.training as training
import torch
import configs.config as cfg_loader

cfg = cfg_loader.get_config()
net = model.NDF()
train_dataset = voxelized_data.VoxelizedDataset('train',
                                          res=cfg.input_res,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_workers=0,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)
val_dataset = voxelized_data.VoxelizedDataset('val',
                                          res=cfg.input_res,
                                          split_file=cfg.split_file,
                                          batch_size=cfg.batch_size,
                                          num_workers=0,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)
trainer = training.Trainer(net,
                           torch.device("cuda"),
                           train_dataset,
                           val_dataset,
                           cfg.exp_dir,
                           optimizer=cfg.optimizer,
                           lr=cfg.lr)
trainer.train_model(cfg.num_epochs)
