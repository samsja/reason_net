import pytest
import torch
from reason_net.data.data_gen import MathDataGenConfig

from reason_net.run import RunConfig, run, omegaconf_to_pydantic
from reason_net.data.data_gen import omegaconf_to_pydantic as omegaconf_to_pydantic_data
from reason_net.data.data_gen import generate as generate_data
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
    conf.data.num_workers = 0
    conf.trainer.wandb.enabled = False

    return conf


@pytest.fixture
def config_data() -> MathDataGenConfig:
    with initialize(
        version_base=None,
        config_path="../reason_net/data/configs",
    ):
        raw_conf = compose(config_name="default.yaml")

        conf = omegaconf_to_pydantic_data(raw_conf)

        conf.size = 100

        return conf


def test_run(config: RunConfig, config_data: MathDataGenConfig, tmp_path):
    config_data.save_file_path = tmp_path / "test.txt"

    generate_data(config_data)

    config.data.dataset_path = config_data.save_file_path

    config.trainer.save_dir = tmp_path
    module, data = run(config)

    if torch.cuda.is_available():
        module.to("cuda")

    idx = torch.Tensor(data.tokenizer.encode("1+1")).long().to(module.device)
    generated = generate(module.model, idx, 10).to("cpu").tolist()
    decoded = data.tokenizer.decode(generated)

    assert decoded is not None
