import pathlib
from argparse import ArgumentParser

import torch
import torchvision
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from torch_vqvae.model import VQVAE

Tensor = torch.Tensor


class BaseExperiment(LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.model = VQVAE(**hparams.model)
        self.hparams = hparams
        self.current_device = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # data args
        parser.add_argument('--data_dir', type=str, default=str(pathlib.Path('./data')))
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=32)
        # optimizer args
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)

        return parser

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, _ = batch

        res = self(image)
        rec_loss = self.model.reconstruction_loss(res.rec, image)
        loss = res.vq_loss + rec_loss

        log = dict(
            vq_loss=res.vq_loss.detach(),
            rec_loss=rec_loss.detach(),
            loss=loss.detach()
        )
        return dict(loss=loss, log=log)

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        res = self(image)
        rec_loss = self.model.reconstruction_loss(res.rec, image)
        loss = res.vq_loss + rec_loss

        out = dict(val_loss=loss)
        if batch_idx == 0:
            out['image'] = image[:10]
            out['rec'] = res.rec[:10]
        return out

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = dict(avg_val_loss=avg_val_loss)

        # log reconstructions
        image = outputs[0]['image']
        rec = outputs[0]['rec']
        grid = torchvision.utils.make_grid(
            torch.cat([image, rec]),
            nrow=image.shape[0], pad_value=0, padding=1,
        )
        self.logger.experiment.add_image('recons', grid, self.current_epoch)

        return {'val_loss': avg_val_loss, 'log': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay,
                                amsgrad=False)

    def prepare_data(self):
        raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)
