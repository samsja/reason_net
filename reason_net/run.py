from typing import Any, cast
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from reason_net.data import MathDataModule, MathDataConfig


class RunConfig(BaseModel):
    seed: int
    data: MathDataConfig


def run(conf: RunConfig):
    data = MathDataModule(conf.data)

    data.prepare_data()
    data.setup("fit")

    print("here")
    print(conf.seed)

    for batch in data.train_dataloader():
        print(batch)


@hydra.main(version_base="1.2")
def main(raw_conf: DictConfig):
    OmegaConf.resolve(raw_conf)
    dict_conf = cast(dict[str, Any], OmegaConf.to_container(raw_conf))
    print(dict_conf)
    conf = RunConfig(**dict_conf)
    run(conf)


if __name__ == "__main__":
    main()
