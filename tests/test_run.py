from pathlib import Path
from omegaconf import DictConfig
import pytest
import torch

from reason_net.run import run, omegaconf_to_pydantic
from reason_net.generate import generate
from hydra import compose, initialize


@pytest.fixture
def raw_config() -> DictConfig:
    with initialize(
        version_base=None,
        config_path="../reason_net/configs",
    ):
        raw_conf = compose(config_name="default.yaml", overrides=["module/model=910K"])

    return raw_conf


def _init_config(cfg: DictConfig, tmp_path: Path) -> DictConfig:
    cfg.trainer.save_dir = tmp_path
    cfg.trainer.lightning.max_epochs = 2
    cfg.data.num_workers = 0
    cfg.wandb.enabled = False
    cfg.data.dataset_path = Path("tests/data-test.txt")
    return cfg


@pytest.mark.parametrize("module_name", ["normal", "reason"])
def test_run(raw_config: DictConfig, tmp_path: Path, module_name: str):
    raw_config = _init_config(raw_config, tmp_path)

    config = omegaconf_to_pydantic(raw_config)
    module, data = run(config)

    if torch.cuda.is_available():
        module.to("cuda")

    idx = torch.Tensor(data.tokenizer.encode("1+1")).long().to(module.device)
    generated = generate(module.model, idx, 10).to("cpu").tolist()
    decoded = data.tokenizer.decode(generated)

    assert decoded is not None
