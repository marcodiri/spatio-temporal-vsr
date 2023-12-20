from lightning.pytorch.cli import LightningCLI

from data.datamodule import FolderDataModule
from models.gan_module import VSRGAN
from models.networks.egvsr_nets import FRNet, SpatioTemporalDiscriminator  # noqa: F401
from utils.lit_callbacks import ImageLog, MemProfiler  # noqa: F401


def cli_main():
    cli = LightningCLI(
        VSRGAN,
        FolderDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
