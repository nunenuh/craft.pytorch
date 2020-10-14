import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics import Accuracy



class TaskCRAFT(LightningModule):
    def __init__(self, model, criterion, optimizer, scheduler=None):
        super(TaskCRAFT, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = Accuracy()

    def forward(self, images, captions):
        predict, feature = self.model(images)
        return predict, feature

    def shared_step(self, batch, batch_idx):
        images, region, affinity = batch

        region = region.type(torch.FloatTensor)
        affinity = affinity.type(torch.FloatTensor)

        images = images.to(self.device)
        region, affinity = region.to(self.device), affinity.to(self.device)

        predict, feature = self.model.forward(images)
        loss = self.criterion(predict, region, affinity)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
#         result = pl.TrainResult(loss)
#         result.log_dict({'trn_loss': loss})
        self.log('trn_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
#         result = pl.EvalResult(checkpoint_on=loss)
#         result.log_dict({'val_loss': loss})
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        return self.optimizer
