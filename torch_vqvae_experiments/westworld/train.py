import pathlib

import torch.backends.cudnn as cudnn
import torchvision
import yaml
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import random_split
from torchvision.transforms import transforms

from torch_vqvae.experiment import BaseExperiment
from torch_vqvae.model import VQVAE


class WestWorldExperiment(BaseExperiment):
    def prepare_data(self):
        data_dir = self.params['data_dir']

        ds = torchvision.datasets.ImageFolder(root=data_dir,
                                              transform=transforms.ToTensor())
        n = len(ds)
        tn = int(n * 0.8)
        vn = n - tn
        self.train_dataset, self.val_dataset = random_split(ds, [tn, vn])


if __name__ == '__main__':
    # For reproducibility
    seed_everything(42)
    cudnn.deterministic = True
    cudnn.benchmark = False

    config_file = pathlib.Path('/Users/bdsaglam/PycharmProjects/torch-vqvae/torch_vqvae_experiments/westworld/config.yaml')
    config_yaml = config_file.read_text()
    config = yaml.safe_load(config_yaml)

    model = VQVAE(**config['model_params'])
    experiment = WestWorldExperiment(model, config['exp_params'])

    runner = Trainer(
        early_stop_callback=False,
        **config['trainer_params']
    )

    runner.fit(experiment)
