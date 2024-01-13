from pathlib import Path
from typing import Literal
from typing_extensions import Self
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, model_validator
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

import torch
import wandb

from reason_net.data import MathDataModule, MathDataConfig
from reason_net.module import NAME_TO_MODULE, NormalModule, ModuleConfig


class PlTrainerConfig(BaseModel):
    precision: Literal["bf16-mixed", "bf16-true", "32-true"]
    max_epochs: int
    log_every_n_steps: int
    devices: int
    val_check_interval: float | int


class WandbConfig(BaseModel):
    enabled: bool
    project_name: str
    name: str | None = None


class TrainerConfig(BaseModel):
    lightning: PlTrainerConfig
    save_dir: Path
    checkpoint_path: Path | None


class RunConfig(BaseModel):
    data: MathDataConfig
    module: ModuleConfig
    module_name: Literal["normal", "reason"]
    trainer: TrainerConfig
    wandb: WandbConfig

    @model_validator(mode="after")
    def reason_net_data(self) -> Self:
        if self.module_name == "reason":
            self.data.dataset_config.reason_net_data = True
        return self


def run(conf: RunConfig) -> tuple[NormalModule, MathDataModule]:
    data = MathDataModule(conf.data)
    module = NAME_TO_MODULE[conf.module_name](conf.module, data.tokenizer)

    if not (conf.wandb.enabled):
        run = wandb.init(mode="disabled")  # type: ignore
    else:
        run = wandb.init(project=conf.wandb.project_name, name=conf.wandb.name)  # type: ignore

    wandb_logger = WandbLogger(save_dir=conf.trainer.save_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=conf.trainer.save_dir / Path(run.name),  # type: ignore
        monitor="val_loss",
        filename="reason_net-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        save_last=True,
    )

    callbacks: list[Callback] = [checkpoint_callback]

    trainer = Trainer(
        **conf.trainer.lightning.model_dump(), logger=wandb_logger, callbacks=callbacks
    )
    ckpt_path = (
        str(conf.trainer.checkpoint_path) if conf.trainer.checkpoint_path else None
    )
    trainer.fit(module, data, ckpt_path=ckpt_path)

    return module, data


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
