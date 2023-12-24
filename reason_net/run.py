from typing import Any, cast
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
import torch

from reason_net.data import MathDataModule, MathDataConfig


class RunConfig(BaseModel):
    seed: int
    data: MathDataConfig


def run(conf: RunConfig):
    data = MathDataModule(conf.data)

    data.prepare_data()
    data.setup("fit")

    print(conf.seed)

    for batch in data.train_dataloader():
        assert len(batch) == 2
        assert len(batch[0]) == len(batch[1])
        assert batch[0].dtype == batch[1].dtype == torch.long


@hydra.main(version_base="1.2")
def main(raw_conf: DictConfig):
    OmegaConf.resolve(raw_conf)
    dict_conf = cast(dict[str, Any], OmegaConf.to_container(raw_conf))
    print(dict_conf)
    conf = RunConfig(**dict_conf)
    run(conf)


if __name__ == "__main__":
    main()
