from pathlib import Path
import random
from typing import Iterable
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from rich.progress import Progress


class MathDataGen:
    operand = ["+", "-", "/", "*", "%"]

    def __init__(self, min: int, max: int):
        self.max = max
        self.min = min

    def generate(self) -> str:
        i = random.randint(0, len(self.operand) - 1)

        operand = self.operand[i]

        left = str(random.randint(10**self.min, 10**self.max))
        right = str(random.randint(10**self.min, 10**self.max))

        if operand == "/":
            code = left + "//" + right
            output = str(eval(code))
            real_code = left + "/" + right + "="
            return real_code + output

        else:
            code = left + operand + right
            output = str(eval(code))
            real_code = left + operand + right + "="
            return real_code + output

    def generate_n(self, n: int) -> Iterable[str]:
        already_generated_hash: list[int] = list()

        with Progress() as progress:
            task_id = progress.add_task("[red]Generating...", total=n)

            while len(already_generated_hash) < n:
                new_data = self.generate()
                new_data_hash = hash(new_data)
                if new_data_hash not in already_generated_hash:
                    already_generated_hash.append(new_data_hash)
                    progress.update(task_id, advance=1)
                    yield new_data


class MathDataGenConfig(BaseModel):
    min: int
    max: int
    size: int
    seed: int
    save_file_path: Path


def generate(conf: MathDataGenConfig):
    random.seed(conf.seed)
    gen = MathDataGen(conf.min, conf.max)
    with open(conf.save_file_path, "w") as f:
        for d in gen.generate_n(conf.size):
            f.write(d + "\n")


def omegaconf_to_pydantic(raw_conf: DictConfig) -> MathDataGenConfig:
    OmegaConf.resolve(raw_conf)
    return MathDataGenConfig(**OmegaConf.to_container(raw_conf))  # type: ignore


@hydra.main(version_base="1.2")
def main(raw_conf: DictConfig):
    conf = omegaconf_to_pydantic(raw_conf)
    print(conf)
    generate(conf)


if __name__ == "__main__":
    main()
