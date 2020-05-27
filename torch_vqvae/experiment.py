import torch
import torchvision
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

Tensor = torch.Tensor


class BaseExperiment(LightningModule):
    def __init__(self,
                 vae_model,
                 params) -> None:
        super().__init__()

        self.model = vae_model
        self.params = params
        self.current_device = None

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, _ = batch

        vq_loss, rec = self(image)[:2]
        rec_loss = self.model.reconstruction_loss(rec, image)
        loss = vq_loss + rec_loss

        log = dict(
            vq_loss=vq_loss.detach(),
            rec_loss=rec_loss.detach(),
            loss=loss.detach()
        )
        return dict(loss=loss, log=log)

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        vq_loss, rec = self(image)[:2]
        rec_loss = self.model.reconstruction_loss(image, rec)
        loss = vq_loss + rec_loss

        out = dict(val_loss=loss)
        if batch_idx == 0:
            out['image'] = image[:10]
            out['rec'] = rec[:10]
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
                                lr=self.params['learning_rate'],
                                weight_decay=self.params['weight_decay'],
                                amsgrad=False)

    def prepare_data(self):
        raise NotImplementedError()

        data_dir = self.params['data_dir']

        ds = torchvision.datasets.ImageFolder(root=data_dir,
                                              transform=transforms.ToTensor())
        n = len(ds)
        tn = int(n * 0.8)
        vn = n - tn
        self.train_dataset, self.val_dataset = random_split(ds, [tn, vn])

    def train_dataloader(self):
        num_workers = self.params.get('num_workers', 1)
        return DataLoader(self.train_dataset,
                          batch_size=self.params['batch_size'],
                          num_workers=num_workers)

    def val_dataloader(self):
        num_workers = self.params.get('num_workers', 1)
        return DataLoader(self.val_dataset,
                          batch_size=self.params['batch_size'],
                          num_workers=num_workers)
