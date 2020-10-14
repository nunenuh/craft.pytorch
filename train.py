import os
import sys

#append this directory to make module work
sys.path.append(os.getcwd())

import argparse
import nonechucks as nc

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl

from craft.datasets import loader
from craft import nn as nnc
from craft.trainer.task import TaskCRAFT
from craft.trainer import helper as trainer_helper
from craft.models.craft import CRAFT

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='trainer craft with synthtext dataset ')
    parser.add_argument('--resume', default=None, type=str,
                        help='Choose pth file to resume training')
    parser.add_argument('--max_epoch', required=True, default=None,
                        type=int, help='How many epoch to run training')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        help='choose learning rate for optimizer, default value is 0.01')
    parser.add_argument('--bsize', '--batch_size', default=8, type=int,
                        help='choose batch size for data loader, default value is 16')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='choose to shuffle data or not, default value is True')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='how many workers to load for running dataset')
    parser.add_argument('--dataset_path', required=True, default='/data/synthtext', type=str,
                        help='path to synthtext dataset')
    parser.add_argument('--image_size', default='224x224', type=str,
                        help='witdh and height of the image, default value is 224x224')
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='fill with zero to use cpu or fill with number 2 to use multigpu')
    parser.add_argument('--log_freq', default=10, type=int,
                        help='show log every value, default value is 10')

    parser.add_argument('--checkpoint_dir', default='saved_checkpoints/', type=str,
                        help='checkpoint directory for saving progress')
    parser.add_argument('--logs_dir', default='logs/', type=str,
                        help='directory logs for tensorboard callback')

    args = parser.parse_args()

    w, h = args.image_size.split('x')
    w, h = int(w), int(h)

    # hyper parameter
    ROOT_PATH = args.dataset_path
    IMSIZE = (w, h)
    BSIZE = args.bsize
    SHUFFLE = args.shuffle
    NWORKERS = args.num_workers

    LRATE = args.lr
    WDECAY = 0.002
    MOMENTUM = 0.9
    SCH_STEP_SIZE = 3
    SCH_GAMMA = 0.1
    
    
    MAX_EPOCHS = args.max_epoch
    NUM_GPUS = args.num_gpus
    LOG_FREQ = args.log_freq 

    SAVED_CHECKPOINT_PATH = args.checkpoint_dir
    SAVED_LOGS_PATH = args.logs_dir
    
    
    CHECKPOINT_RESUME = False
    CHECKPOINT_PATH = None
    
    WEIGHT_RESUME = False
    WEIGHT_PATH = None
    
    if args.resume:
        fpath = Path(args.resume)
        if fpath.is_file():
            if fpath.suffix == 'ckpt':
                # it means checkpoint of pytorch lightning 
                CHECKPOINT_RESUME = True
                CHECKPOINT_PATH = str(fpath)
            elif fpath.suffix == 'pth':
                # it means pytorch file original from model
                WEIGHT_RESUME = True
                WEIGHT_PATH = str(fpath)       
            else:
                raise NotImplemented(f'File with {fpath.suffix} is not implemented! ' 
                                     f'make sure you load valid file with ckpt or pth extension!')
        else:
            raise IOError(f'Path that you specified is not valid pytorch or pytorch-lighning path!')
        
    
    
    

    # trailoader and validloader
    trainloader = loader.synthtext_trainloader(path=ROOT_PATH, batch_size=BSIZE,
                                               shuffle=SHUFFLE, nworkers=NWORKERS)

    validloader = loader.synthtext_validloader(path=ROOT_PATH, batch_size=BSIZE,
                                               shuffle=False, nworkers=NWORKERS)
    
    
    # Model Preparation
    if WEIGHT_RESUME:
        model = CRAFT(pretrained=True)
        weights = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))
        weights = trainer_helper.copy_state_dict(weights)
        model.load_state_dict(weights)
        trainer_helper.freeze_network(model)
        trainer_helper.unfreeze_conv_cls_module(model)
    else:
        model = CRAFT(pretrained=True)
        
        
    criterion = nnc.OHEMLoss()
    optimizer = optim.SGD(model.parameters(), lr=LRATE, weight_decay=WDECAY, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCH_STEP_SIZE, gamma=SCH_GAMMA)
        
    
    if CHECKPOINT_RESUME:
        task = TaskCRAFT.load_from_checkpoint(CHECKPOINT_PATH)
    else:
        task = TaskCRAFT(model, criterion, optimizer, scheduler)

    
    # DEFAULTS used by the Trainer
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath=SAVED_CHECKPOINT_PATH,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='craft_net_'
    )
    tensorboard_logger = pl.loggers.TensorBoardLogger(SAVED_LOGS_PATH)

    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, gpus=NUM_GPUS,
                         logger=tensorboard_logger,
                         checkpoint_callback=model_checkpoint,
                         log_every_n_steps=LOG_FREQ,
                         num_sanity_val_steps=0)

    # start training the model
    trainer.fit(task, trainloader, validloader)
