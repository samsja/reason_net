import pytest

from reason_net.run import RunConfig, run, omegaconf_to_pydantic
from hydra import compose, initialize


@pytest.fixture
def config() -> RunConfig:
    with initialize(
        version_base=None,
        config_path="../reason_net/configs",
    ):
        raw_conf = compose(config_name="default.yaml", overrides=["module/model=2M"])

    conf = omegaconf_to_pydantic(raw_conf)

    conf.trainer.lightning.max_epochs = 2
    conf.data.size = 16

    return conf


def test_run(config: RunConfig):
    run(config)