import random
from typing import ClassVar, TypeAlias, TypeVar

from pydantic import BaseModel
import torch
from torch import Tensor
import lightning as L
from torch.utils.data import Dataset, random_split, DataLoader
from jaxtyping import Int


class MathDataGen:
    operand = ["+", "-", "/", "*", "%"]

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
            real_code = left + "/" + right + "="
            return real_code, output

        else:
            code = left + operand + right
            output = str(eval(code))
            real_code = left + operand + right + "="
            return real_code, output


class MathTokenizer:
    max_digit = 10
    pad_token = "P"
    eos_token = "S"
    unknown_token = "U"
    operand = MathDataGen.operand
    equal_token = "="

    def __init__(self) -> None:
        digits = [str(i) for i in range(self.max_digit)]
        self.anti_vocab: list[str] = (
            digits
            + [op for op in self.operand]
            + [self.pad_token, self.eos_token, self.equal_token]
        )
        self.vocab = {term: i for i, term in enumerate(self.anti_vocab)}

    def encode(self, x: str) -> list[int]:
        return list(map(lambda x: self.vocab[x], list(x)))

    def _decode_caract(self, x: int) -> str:
        if x < self.vocab_size:
            return self.anti_vocab[x]
        else:
            return self.unknown_token

    def decode(self, x: list[int]) -> str:
        decoded = list(map(lambda x: self._decode_caract(x), list(x)))
        return "".join(decoded)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


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
    num_workers: int


seq = TypeVar("seq")
b = TypeVar("b")

DataPoint: TypeAlias = tuple[Int[Tensor, "seq"], tuple[int, int]]
BatchDataPoint: TypeAlias = tuple[Int[Tensor, "b seq"], list[Int[Tensor, "b"]]]


class MathDataModule(L.LightningDataModule):
    train_prop: ClassVar[float] = 0.8

    def __init__(self, conf: MathDataConfig):
        super().__init__()
        self.conf = conf

    def prepare_data(self) -> None:
        self.generator = MathDataGen(self.conf.min, self.conf.max)
        self.tokenizer = MathTokenizer()

    def _generate_data_point(self) -> tuple[list[int], tuple[int, int]]:
        exo, resp = self.generator.generate()
        exo_tokenized = self.tokenizer.encode(exo)
        resp_tokenized = self.tokenizer.encode(resp)

        data = exo_tokenized + resp_tokenized + [self.tokenizer.eos_token_id]

        start = len(exo_tokenized)
        end = len(data)

        return data, (start, end)

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders

        if stage != "fit":
            raise NotImplementedError(f"DataModule stage {stage} not implemented")

        all_raw_data = [self._generate_data_point() for _ in range(self.conf.size)]

        all_data: list[DataPoint] = []

        max_len = max([len(data) for data, _ in all_raw_data])

        for data, start_end in all_raw_data:
            data += [self.tokenizer.pad_token_id] * (max_len - len(data))
            all_data.append((torch.tensor(data).long(), start_end))

        self.dataset = ListDataset(all_data)

        train_size = int(self.train_prop * len(self.dataset))
        val_size = len(self.dataset) - train_size

        self.train, self.val = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.conf.seed),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.conf.batch_size, num_workers=self.conf.num_workers
        )
