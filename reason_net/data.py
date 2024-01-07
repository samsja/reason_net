from pathlib import Path
from typing import ClassVar, TypeAlias, TypeVar

from pydantic import BaseModel
import torch
from torch import Tensor
import lightning as L
from torch.utils.data import Dataset, random_split, DataLoader
from jaxtyping import Int, jaxtyped
from beartype import beartype as typechecker
import numpy as np


class MathTokenizer:
    max_digit = 10
    pad_token = "P"
    eos_token = "S"
    unknown_token = "U"
    operand = ["+", "-", "/", "*", "%"]
    equal_token = "="

    def __init__(self) -> None:
        digits = [str(i) for i in range(self.max_digit)]
        self.anti_vocab: list[str] = (
            [self.pad_token, self.eos_token, self.equal_token]
            + digits
            + [op for op in self.operand]
        )
        self.vocab = {term: i for i, term in enumerate(self.anti_vocab)}

    def _encode_caract(self, x: str) -> int:
        if x in self.vocab:
            return self.vocab[x]
        else:
            return self.vocab[self.unknown_token]

    def encode(self, x: str) -> list[int]:
        return list(map(lambda x: self.vocab[x], list(x)))

    def _decode_caract(self, x: int) -> str:
        if x < self.vocab_size:
            return self.anti_vocab[x]
        else:
            return self.unknown_token

    def decode(self, x: list[int]) -> str:
        decoded = [self._decode_caract(i) for i in x]
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


class MathDataConfig(BaseModel):
    seed: int
    batch_size: int
    num_workers: int
    dataset_path: Path


seq = TypeVar("seq")
b = TypeVar("b")

DataPoint: TypeAlias = tuple[Int[Tensor, "seq"], int]
BatchDataPoint: TypeAlias = Int[Tensor, "b seq"]


class MathDataset(Dataset):
    def __init__(self, dataset_path: Path, tokenizer: MathTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        raw_data = list()
        with open(dataset_path, "r") as f:
            for line in f:
                raw_data.append(line.strip())

        self.data = np.array(raw_data)
        # seeh her why np array and not list https://docs.aws.amazon.com/codeguru/detector-library/python/pytorch-data-loader-with-multiple-workers/ # noqa: E501

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> list[int]:
        data_point = self.data[idx]
        data = self.tokenizer.encode(data_point) + [self.tokenizer.eos_token_id]
        return data


class DataCollatorLangModeling:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    @jaxtyped(typechecker=typechecker)
    def collate_batch(self, batch: list[list[int]]) -> Int[Tensor, "b seq"]:
        max_length = max(len(item) for item in batch)

        padded_batch = []

        for item in batch:
            num_padding = max_length - len(item)

            padded_sequence = item + [self.pad_token_id] * num_padding
            padded_batch.append(padded_sequence)

        return torch.tensor(padded_batch)


class MathDataModule(L.LightningDataModule):
    train_prop: ClassVar[float] = 0.8

    def __init__(self, conf: MathDataConfig):
        super().__init__()
        self.conf = conf

    def prepare_data(self) -> None:
        self.tokenizer = MathTokenizer()
        self.dataset = MathDataset(self.conf.dataset_path, self.tokenizer)
        self.data_collator = DataCollatorLangModeling(self.tokenizer.pad_token_id)

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders

        if stage != "fit":
            raise NotImplementedError(f"DataModule stage {stage} not implemented")

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
            collate_fn=self.data_collator.collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
            collate_fn=self.data_collator.collate_batch,
        )
