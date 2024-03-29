from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import os


class DANNDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.mnistm_url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        from six.moves import urllib
        import gzip

        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

        # Check if MNIST-M has already been downloaded
        if os.path.exists(os.path.join(self.hparams.data_dir, self.training_file)) and os.path.exists(
            os.path.join(self.hparams.data_dir, self.test_file)):
            return

        # download pkl files
        print("Downloading " + self.mnistm_url)
        filename = self.mnistm_url.rpartition("/")[2]
        file_path = os.path.join(self.hparams.data_dir, filename)
        if not os.path.exists(file_path.replace(".gz", "")):
            data = urllib.request.urlopen(self.mnistm_url)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            mnist_train_labels = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            mnist_test_labels = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)

            mnist_m_train_data
            mnist_m_test_data




            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        # return DataLoader(
        #     dataset=self.data_train,
        #     batch_size=self.hparams.batch_size,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     shuffle=True,
        # )
        src_loader = DataLoader(self.train_set_src, batch_size=self.cfg.training.batch_size, shuffle=True,
                                num_workers=self.cfg.training.num_workers, pin_memory=True, sampler=None,
                                drop_last=True)
        tgt_loader = DataLoader(self.train_set_tgt, batch_size=self.cfg.training.batch_size, shuffle=True,
                                num_workers=self.cfg.training.num_workers, pin_memory=True, sampler=None,
                                drop_last=True)

        self.len_dataloader = min(len(src_loader), len(tgt_loader))
        return list(zip(src_loader, tgt_loader))

    def val_dataloader(self):
        # return DataLoader(
        #     dataset=self.data_val,
        #     batch_size=self.hparams.batch_size,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     shuffle=False,
        # )
        return DataLoader(self.val_set_src, batch_size=self.cfg.training.batch_size,
                          num_workers=self.cfg.training.num_workers)

    def test_dataloader(self):
        # return DataLoader(
        #     dataset=self.data_test,
        #     batch_size=self.hparams.batch_size,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     shuffle=False,
        # )
        return DataLoader(self.test_set_tgt, batch_size=self.cfg.training.batch_size,
                          num_workers=self.cfg.training.num_workers)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
