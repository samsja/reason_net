import random
from typing import Any, ClassVar

from pydantic import BaseModel
import torch
import lightning as L
from torch.utils.data import Dataset, random_split, DataLoader


class MathDataGen:
    operand = ["+", "/", "*", "%"]

    def __init__(self, min: int, max: int):
        self.max = max
        self.min = min

    def generate(self) -> tuple[str, str]:
        i = random.randint(0, len(self.operand) - 1)

        operand = self.operand[i]

        left = str(random.randint(10**self.min, 10**self.max))
        right = str(random.randint(10**self.min, 10**self.max))

        if operand == "/":
            code = left + "//" + right
            output = str(eval(code))
            real_code = left + "/" + right
            return real_code, output

        else:
            code = left + operand + right
            return code, str(eval(code))


class MathTokenizer:
    max_digit = 10
    pad_token = "P"

    def __init__(self, operand: list[str]):
        digits = [str(i) for i in range(self.max_digit)]
        self.anti_vocab: list[str] = digits + [op for op in operand] + [self.pad_token]
        self.vocab = {term: i for i, term in enumerate(self.anti_vocab)}

    def encode(self, x: str) -> list[int]:
        return list(map(lambda x: self.vocab[x], list(x)))

    def decode(self, x: list[int]) -> str:
        decoded = list(map(lambda x: self.anti_vocab[x], list(x)))
        return "".join(decoded)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MathDataConfig(BaseModel):
    min: int
    max: int
    size: int
    seed: int
    batch_size: int


class MathDataModule(L.LightningDataModule):
    train_prop: ClassVar[float] = 0.8

    def __init__(self, conf: MathDataConfig):
        self.conf = conf

    def prepare_data(self) -> None:
        self.generator = MathDataGen(self.conf.min, self.conf.max)
        self.tokenizer = MathTokenizer(MathDataGen.operand)

    def _generate_data_point(self) -> tuple[list[int], int]:
        exo, resp = self.generator.generate()
        exo_tokenized = self.tokenizer.encode(exo)
        resp_tokenized = self.tokenizer.encode(resp)

        return exo_tokenized + resp_tokenized, len(exo_tokenized)

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders

        if stage != "fit":
            raise NotImplementedError(f"DataModule stage {stage} not implemented")

        all_data_point: list[Any] = [
            self._generate_data_point() for _ in range(self.conf.size)
        ]

        max_len = max([len(data) for data, _ in all_data_point])

        for i, (data, cutoff) in enumerate(all_data_point):
            data += [self.tokenizer.pad_token_id] * (max_len - len(data))
            all_data_point[i] = (torch.tensor(data).long(), cutoff)

        self.dataset = ListDataset(all_data_point)

        train_size = int(self.train_prop * len(self.dataset))
        val_size = len(self.dataset) - train_size

        self.train, self.val = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.conf.seed),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.conf.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.conf.batch_size)
