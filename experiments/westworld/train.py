import pathlib
from argparse import Namespace, ArgumentParser

import torch.backends.cudnn as cudnn
import torchvision
import yaml
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import random_split
from torchvision.transforms import transforms

from torch_vqvae.experiment import BaseExperiment


class WestWorldExperiment(BaseExperiment):
    def prepare_data(self):
        ds = torchvision.datasets.ImageFolder(root=self.hparams.data_dir,
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
    experiment = WestWorldExperiment(Namespace(**hparams))

    # prepare trainer params
    trainer_params = vars(args)
    del trainer_params['hparams_file']
    runner = Trainer(**trainer_params)

    runner.fit(experiment)
