from typing import Any, cast
import hydra
from omegaconf import DictConfig, OmegaConf
from reason_net.data import MathDataModule
from pydantic import BaseModel


class RunConfig(BaseModel):
    seed: int


def run(conf: RunConfig):
    _data = MathDataModule(1, 2, 1000, conf.seed)

    print("here")
    print(conf.seed)


@hydra.main(version_base="1.2")
def main(raw_conf: DictConfig):
    dict_conf = cast(dict[str, Any], OmegaConf.to_container(raw_conf))
    conf = RunConfig(**dict_conf)
    run(conf)


if __name__ == "__main__":
    main()
