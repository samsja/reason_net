from pathlib import Path
from omegaconf import open_dict
import pytest
import torch

from reason_net.run import RunConfig, run, omegaconf_to_pydantic
from reason_net.generate import generate
from hydra import compose, initialize


def _init_config() -> RunConfig:
    with initialize(
        version_base=None,
        config_path="../reason_net/configs",
    ):
        cfg = compose(config_name="all-14m.yaml", overrides=["module/model=910K"])
    with open_dict(cfg):
        cfg.trainer.pl.max_epochs = 2
        cfg.data.num_workers = 0
        cfg.wandb.enabled = False
        cfg.data.dataset_path = Path("tests/data-test.txt")

        cfg.trainer.callbacks = {
            "norm_monitor": {"log_every_n_steps": 1},
        }

    return omegaconf_to_pydantic(cfg)


def normal_config() -> RunConfig:
    config = _init_config()

    config.reason_mode = False

    return config


def reason_middle_config() -> RunConfig:
    config = _init_config()

    config.reason_mode = True
    config.data.reason.reason_token_pos = "middle"  # type: ignore

    return config


def reason_left_config() -> RunConfig:
    config = _init_config()

    config.reason_mode = True
    config.data.reason.reason_token_pos = "left"  # type: ignore

    return config


@pytest.mark.parametrize(
    "config", [normal_config(), reason_middle_config(), reason_left_config()]
)
def test_run(config: RunConfig, tmp_path: Path):
    config.trainer.save_dir = tmp_path

    module, data = run(config)

    if torch.cuda.is_available():
        module.to("cuda")

    idx = torch.Tensor(data.tokenizer.encode("1+1")).long().to(module.device)
    generated = generate(module.model, idx, 10).to("cpu").tolist()
    decoded = data.tokenizer.decode(generated)

    assert decoded is not None
