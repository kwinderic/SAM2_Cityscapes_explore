import os
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from hydra.utils import instantiate
from training.trainer import Trainer
from training.utils.train_utils import makedir, register_omegaconf_resolvers

parser = argparse.ArgumentParser(description="Fine-tune SAM2 on Cityscapes")
parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file")
parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
args = parser.parse_args()

# Initialize Hydra configuration
initialize_config_module(config_module="sam2", version_base="1.2")
register_omegaconf_resolvers()
cfg = compose(config_name=args.config)

# import pdb
# pdb.set_trace()

# Flatten the configuration structure
cfg = cfg.configs.sam2['1_training']

# Access configuration directly
launcher_cfg = cfg.launcher
trainer_cfg = cfg.trainer

# Data preparation
transform = transforms.Compose([
    transforms.Resize((cfg.scratch.resolution, cfg.scratch.resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(cfg.dataset.img_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=cfg.scratch.train_batch_size, shuffle=True, num_workers=cfg.scratch.num_train_workers)

val_dataset = datasets.ImageFolder(cfg.dataset.gt_folder, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=cfg.scratch.train_batch_size, shuffle=False, num_workers=cfg.scratch.num_train_workers)

# Model, criterion, optimizer
model = instantiate(cfg.model).cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg.scratch.base_lr)

# Trainer
import pdb
pdb.set_trace()
trainer = instantiate(trainer_cfg, data={"train": train_loader, "val": val_loader}, model=model, optim=optimizer, loss=criterion)

trainer.run()