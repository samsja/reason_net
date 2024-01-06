import pytest
import torch

from reason_net.run import RunConfig, run, omegaconf_to_pydantic
from reason_net.generate import generate
from hydra import compose, initialize


@pytest.fixture
def config() -> RunConfig:
    with initialize(
        version_base=None,
        config_path="../reason_net/configs",
    ):
        raw_conf = compose(config_name="default.yaml", overrides=["module/model=910K"])

    conf = omegaconf_to_pydantic(raw_conf)

    conf.trainer.lightning.max_epochs = 2
    conf.data.size = 16
    conf.data.num_workers = 0
    conf.trainer.wandb.enabled = False

    return conf


def test_run(config: RunConfig, tmp_path):
    config.trainer.save_dir = tmp_path
    module, data = run(config)

    if torch.cuda.is_available():
        module.to("cuda")

    idx = torch.Tensor(data.tokenizer.encode("1+1")).long().to(module.device)
    generated = generate(module.model, idx, 10).to("cpu").tolist()
    decoded = data.tokenizer.decode(generated)

    assert decoded is not None
