from typing import Literal
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from reason_net.data import MathDataModule, MathDataConfig
from reason_net.model import LLamModule, ModuleConfig


class PlTrainerConfig(BaseModel):
    precision: Literal["bf16-mixed", "bf16-true"]
    max_epochs: int
    log_every_n_steps: int
    devices: int


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
        WandbLogger(project=conf.trainer.wandb.project_name)
        if conf.trainer.wandb.enabled
        else None
    )

    trainer = Trainer(
        accelerator="gpu", **conf.trainer.lightning.model_dump(), logger=wandb_logger
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
    main()
