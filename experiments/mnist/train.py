import pathlib
from argparse import Namespace, ArgumentParser

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

    parser = ArgumentParser()
    parser.add_argument('--hparams_file', type=str, default=None)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare hparams
    if args.hparams_file is not None:
        hparams_file = pathlib.Path(args.hparams_file)
    else:
        script_file = pathlib.Path(__file__).absolute()
        hparams_file = script_file.parent / 'hparams.yaml'
    hparams = yaml.safe_load(hparams_file.read_text())
    experiment = MNISTExperiment(Namespace(**hparams))

    # prepare trainer params
    trainer_params = vars(args)
    del trainer_params['hparams_file']
    runner = Trainer(**trainer_params)

    runner.fit(experiment)
