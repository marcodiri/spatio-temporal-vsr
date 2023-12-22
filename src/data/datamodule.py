import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split

from data.folder_dataset import FolderDataset


class FolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        hr_path,
        lr_path="",
        extension="jpg",
        *,
        patch_size,
        tempo_extent=10,
        hr_path_filter="",
        lr_path_filter="",
        dataset_upscale_factor=2,
        rescale_factor=None,
        train_pct=0.8,
        batch_size=32,
    ):
        """
        Custom PyTorch Lightning DataModule.

        See :class:`~folder_dataset.FolferDataset` for details on args.

        Args
            train_pct (float):
                Percentage of the training data to use as validation.
            batch_size (int):
                Size of every training batch.
        """

        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = FolderDataset(**self.hparams)
            train_set_size = int(len(dataset) * self.hparams.train_pct)
            valid_set_size = len(dataset) - train_set_size

            # split the train set into two
            self.train_set, self.valid_set = random_split(
                dataset, [train_set_size, valid_set_size]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=12,
            shuffle=True,
            pin_memory=True,
        )
        return data_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        data_loader_eval = DataLoader(
            dataset=self.valid_set,
            batch_size=self.hparams.batch_size,
            num_workers=12,
            shuffle=False,
            pin_memory=True,
        )
        return data_loader_eval
