import pathlib
from argparse import Namespace

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

    # prepare config
    script_file = pathlib.Path(__file__).absolute()
    config_file = script_file.parent / 'config.yaml'
    config = yaml.safe_load(config_file.read_text())
    hparams = dict(
        model_params=config['model_params'],
        **config['experiment_params'],
        **config['trainer_params'],
    )

    experiment = WestWorldExperiment(Namespace(**hparams))

    runner = Trainer(
        early_stop_callback=False,
        overfit_pct=0.1,
        **config['trainer_params'])

    runner.fit(experiment)
