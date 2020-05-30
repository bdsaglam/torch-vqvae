import pathlib

import torch.backends.cudnn as cudnn
import yaml
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from torch_vqvae.experiment import BaseExperiment
from torch_vqvae.model import VQVAE


class MNISTExperiment(BaseExperiment):
    def prepare_data(self):
        data_dir = self.params['data_dir']

        # transform
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # download
        mnist_train = MNIST(data_dir, train=True, download=True, transform=transform)

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in data loaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val


if __name__ == '__main__':
    # For reproducibility
    seed_everything(42)
    cudnn.deterministic = True
    cudnn.benchmark = False

    config_file = pathlib.Path('/Users/bdsaglam/PycharmProjects/torch-vqvae/torch_vqvae_experiments/mnist/config.yaml')
    config_yaml = config_file.read_text()
    config = yaml.safe_load(config_yaml)

    model = VQVAE(**config['model_params'])
    experiment = MNISTExperiment(model, config['exp_params'])

    runner = Trainer(
        early_stop_callback=False,
        overfit_pct=0.1,
        **config['trainer_params'])

    runner.fit(experiment)
