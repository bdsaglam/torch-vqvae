import pathlib
from argparse import Namespace

import torch.backends.cudnn as cudnn
import yaml
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from torch_vqvae.experiment import BaseExperiment


class MNISTExperiment(BaseExperiment):
    def prepare_data(self):
        # transform
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # download
        mnist_train = MNIST(self.hparams.data_dir, train=True, download=True,
                            transform=transform)

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

    # prepare config
    script_file = pathlib.Path(__file__).absolute()
    config_file = script_file.parent / 'config.yaml'
    config = yaml.safe_load(config_file.read_text())
    hparams = dict(
        model_params=config['model_params'],
        **config['experiment_params'],
        **config['trainer_params'],
    )

    experiment = MNISTExperiment(Namespace(**hparams))

    runner = Trainer(
        early_stop_callback=False,
        overfit_pct=0.1,
        **config['trainer_params'])

    runner.fit(experiment)
