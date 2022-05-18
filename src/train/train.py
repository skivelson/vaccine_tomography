import os, sys
import h5py
import torch
import numpy as np
from tqdm import tqdm

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPSpawnPlugin
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger

from models import UNet3D
from datasets import TomoDataset

sys.path.append('..')
import config
import arg_parser


def main():
    args = arg_parser.get_args(sys.argv[0])
    
    train_loader = DataLoader(
        TomoDataset(os.path.join(config.TRAIN_TOMO_DIR, args.data)),
        batch_size=1,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )

    model = UNet3D([int(c) for c in args.channels])
    logger = CSVLogger(config.LOG_DIR, args.exp_name, flush_logs_every_n_steps=10)
    logger.log_hyperparams(vars(args))

    trainer = Trainer(
        gpus=1,
        #precision=16,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        logger=logger
        # strategy=DDPSpawnPlugin(find_unused_parameters=False)
    )
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
