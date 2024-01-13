from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, TypeAlias, TypeVar, TypedDict

from pydantic import BaseModel, ConfigDict
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
    reason_token = "R"

    def __init__(self) -> None:
        digits = [str(i) for i in range(self.max_digit)]
        self.anti_vocab: list[str] = (
            [self.pad_token, self.eos_token, self.equal_token, self.reason_token]
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
    def reason_token_id(self) -> int:
        return self.vocab[self.reason_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class MathDatasetConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    reason_net_data: bool = False
    reason_net_token_num: int


class MathDataConfig(BaseModel):
    seed: int
    batch_size: int
    num_workers: int
    dataset_path: Path

    dataset_config: MathDatasetConfig


seq = TypeVar("seq")
b = TypeVar("b")


class MathOutputDict(TypedDict):
    cutoff: int
    reason_token_num: int


class MathOutputDictBatch(TypedDict):
    cutoff: list[int]
    reason_token_num: list[int]


MathDataSetOutput: TypeAlias = tuple[list[int], MathOutputDict]


class MathDataset(Dataset):
    def __init__(
        self, dataset_path: Path, tokenizer: MathTokenizer, config: MathDatasetConfig
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        raw_data = list()
        with open(dataset_path, "r") as f:
            for line in f:
                raw_data.append(line.strip())

        self.data = np.array(raw_data)
        # seeh her why np array and not list https://docs.aws.amazon.com/codeguru/detector-library/python/pytorch-data-loader-with-multiple-workers/ # noqa: E501

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> MathDataSetOutput:
        data_point: str = self.data[idx]

        [left, right] = data_point.split("=")

        data_left = self.tokenizer.encode(left)
        data_right = self.tokenizer.encode(right)

        if self.config.reason_net_data:
            data = (
                data_left
                + [self.tokenizer.reason_token_id] * self.config.reason_net_token_num
                + data_right
                + [self.tokenizer.eos_token_id]
            )
            return data, {
                "cutoff": len(data_left),
                "reason_token_num": self.config.reason_net_token_num,
            }
        else:
            data = data_left + data_right + [self.tokenizer.eos_token_id]
            return data, {"cutoff": len(data_left), "reason_token_num": 0}


BatchDataPoint: TypeAlias = tuple[Int[Tensor, "b seq"], MathOutputDictBatch]


class DataCollatorLangModeling:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    @jaxtyped(typechecker=typechecker)
    def collate_batch(self, batch: list[MathDataSetOutput]) -> BatchDataPoint:
        max_length = max(len(item) for item, _ in batch)

        padded_batch = []

        dict_data: MathOutputDictBatch = defaultdict(list)  # type: ignore

        for item, dict_item in batch:
            num_padding = max_length - len(item)

            padded_sequence = item + [self.pad_token_id] * num_padding
            padded_batch.append(padded_sequence)

            for key in dict_item.keys():
                dict_data[key].append(dict_item[key])  # type: ignore

        return torch.tensor(padded_batch), dict_data


class MathDataModule(L.LightningDataModule):
    train_prop: ClassVar[float] = 0.8

    def __init__(self, conf: MathDataConfig):
        super().__init__()
        self.conf = conf
        self.tokenizer = MathTokenizer()

    def prepare_data(self) -> None:
        self.dataset = MathDataset(
            self.conf.dataset_path, self.tokenizer, self.conf.dataset_config
        )
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
