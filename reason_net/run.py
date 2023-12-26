from typing import Literal
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch

from reason_net.data import MathDataModule, MathDataConfig
from reason_net.module import LLamModule, ModuleConfig


class PlTrainerConfig(BaseModel):
    precision: Literal["bf16-mixed", "bf16-true"]
    max_epochs: int
    log_every_n_steps: int
    devices: int
    val_check_interval: float | int


class WandbConfig(BaseModel):
    enabled: bool
    project_name: str


class TrainerConfig(BaseModel):
    lightning: PlTrainerConfig
    wandb: WandbConfig


class RunConfig(BaseModel):
    data: MathDataConfig
    module: ModuleConfig
    trainer: TrainerConfig


def run(conf: RunConfig):
    data = MathDataModule(conf.data)
    module = LLamModule(conf.module)

    wandb_logger = (
        WandbLogger(project=conf.trainer.wandb.project_name, save_dir="lightning_logs")
        if conf.trainer.wandb.enabled
        else None
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="reason_net-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        save_last=True,
    )

    callbacks = [checkpoint_callback]

    trainer = Trainer(
        **conf.trainer.lightning.model_dump(), logger=wandb_logger, callbacks=callbacks
    )

    trainer.fit(module, data)


def omegaconf_to_pydantic(raw_conf: DictConfig) -> RunConfig:
    OmegaConf.resolve(raw_conf)
    return RunConfig(**OmegaConf.to_container(raw_conf))  # type: ignore


@hydra.main(version_base="1.2")
def main(raw_conf: DictConfig):
    conf = omegaconf_to_pydantic(raw_conf)
    print(conf)
    run(conf)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
