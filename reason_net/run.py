from pathlib import Path
from typing import Literal
from typing_extensions import Self
import hydra
from omegaconf import DictConfig, OmegaConf
from reason_net.calllbacks.norm_monitor import NormMonitorConfig
from reason_net.calllbacks.perf_monitor import PerfMonitorConfig
from reason_net.pydantic_conf import Config
from pydantic import model_validator
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor


import torch
import wandb

from reason_net.data import MathDataModule, MathDataConfig, ReasonConfig
from reason_net.module import LLaMaModule, ModuleConfig, ReasonModuleConfig


class PlTrainerConfig(Config):
    precision: Literal["bf16-mixed", "bf16-true", "32-true"] = "bf16-mixed"
    max_epochs: int = 1
    log_every_n_steps: int = 1
    devices: int
    val_check_interval: int | float = 1.0
    gradient_clip_val: float | None = 1.0


class WandbConfig(Config):
    enabled: bool
    project_name: str
    name: str | None = None


class CallbackConfig(Config):
    norm_monitor: NormMonitorConfig | None = None
    perf_monitor: PerfMonitorConfig | None = None


class TrainerConfig(Config):
    pl: PlTrainerConfig
    save_dir: Path
    checkpoint_path: Path | None = None
    callbacks: CallbackConfig = CallbackConfig()


class RunConfig(Config):
    data: MathDataConfig
    module: ModuleConfig
    reason_mode: bool
    trainer: TrainerConfig
    wandb: WandbConfig

    @model_validator(mode="after")
    def reason_mode_setup(self) -> Self:
        if self.reason_mode:
            if self.data.reason is None:
                self.data.reason = ReasonConfig()
                # this put default reason value in case none are passed

            if self.module.reason is None:
                self.module.reason = ReasonModuleConfig(
                    reason_token_num=self.data.reason.reason_token_num
                )
            else:
                self.module.reason.reason_token_num = self.data.reason.reason_token_num
        else:
            self.data.reason = None
            self.module.reason = None

        return self


def run(conf: RunConfig) -> tuple[LLaMaModule, MathDataModule]:
    data = MathDataModule(conf.data)
    module = LLaMaModule(conf.module, data.tokenizer)

    if not (conf.wandb.enabled):
        _ = wandb.init(mode="disabled")  # type: ignore

    wandb_logger = WandbLogger(
        save_dir=conf.trainer.save_dir,
        project=conf.wandb.project_name,
        name=conf.wandb.name,
    )

    wandb_logger.log_hyperparams(conf.model_dump())

    save_sub_path = (
        Path(wandb_logger._experiment.name)
        if wandb_logger._experiment
        else Path("undefined")
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=conf.trainer.save_dir / save_sub_path,  # type: ignore
        monitor="val_loss",
        filename="reason_net-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        save_last=True,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    callbacks: list[Callback] = [checkpoint_callback, lr_monitor_callback]

    for _, callback_conf in conf.trainer.callbacks:
        if callback_conf is not None:
            callbacks.append(callback_conf._target_(callback_conf))

    trainer = Trainer(
        **conf.trainer.pl.model_dump(), logger=wandb_logger, callbacks=callbacks
    )
    ckpt_path = (
        str(conf.trainer.checkpoint_path) if conf.trainer.checkpoint_path else None
    )
    trainer.fit(module, data, ckpt_path=ckpt_path)

    trainer.test(module, datamodule=data)
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
