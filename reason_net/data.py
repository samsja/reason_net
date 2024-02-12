from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, TypeAlias, TypeVar
import typing

from reason_net.pydantic_conf import Config
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
    def equal_token_id(self) -> int:
        return self.vocab[self.equal_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class MathDataConfig(Config):
    seed: int = 42
    batch_size: int
    num_workers: int
    dataset_path: Path
    reason_net_data: bool = False
    reason_net_token_num: int = 20


seq = TypeVar("seq")
b = TypeVar("b")


DatasetOutput: TypeAlias = tuple[list[int], dict[str, int]]


class MathDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        tokenizer: MathTokenizer,
    ):
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

    def __getitem__(self, idx) -> DatasetOutput:
        data_point: str = self.data[idx]

        [left, right] = data_point.split("=")

        left = left

        data_left = self.tokenizer.encode(left)
        data_right = self.tokenizer.encode(right)

        data = (
            data_left
            + [self.tokenizer.equal_token_id]  # add back the equal token
            + data_right
            + [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        )
        # adding pad token here so that the eos is taken into account in the input
        return data, {"cutoff": len(data_left)}


class MathDatasetReason(MathDataset):
    def __init__(
        self, dataset_path: Path, tokenizer: MathTokenizer, reason_net_token_num: int
    ):
        self.reason_net_token_num = reason_net_token_num
        super().__init__(dataset_path, tokenizer)

    def __getitem__(self, idx) -> DatasetOutput:
        data_point: str = self.data[idx]

        [left, right] = data_point.split("=")

        data_left = self.tokenizer.encode(left)
        data_right = self.tokenizer.encode(right)

        data = (
            data_left
            + [self.tokenizer.equal_token_id]  # add back the equal token
            + [self.tokenizer.reason_token_id] * self.reason_net_token_num
            + data_right
            + [self.tokenizer.eos_token_id]
        )
        return data, {
            "cutoff": len(data_left),
            "reason_net_token_num": self.reason_net_token_num,
        }


BatchDataPoint: TypeAlias = tuple[Int[Tensor, "b seq"], Int[Tensor, "b seq_minus_one"]]


class DataCollatorLangModeling:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    @jaxtyped(typechecker=typechecker)
    def collate_batch(self, batch: list[DatasetOutput]) -> BatchDataPoint:
        max_length = max(len(item) for item, _ in batch)

        padded_batch = []

        dict_data = defaultdict(list)

        for item, dict_item in batch:
            num_padding = max_length - len(item)

            padded_sequence = item + [self.pad_token_id] * num_padding
            padded_batch.append(padded_sequence)

            for key in dict_item.keys():
                dict_data[key].append(dict_item[key])

        padded_batch_tensor = torch.tensor(padded_batch)

        target = padded_batch_tensor[:, 1:].clone()

        for b, (_, d_data) in enumerate(batch):
            cutoff = d_data["cutoff"]
            target[b, 0:cutoff] = self.pad_token_id

        return padded_batch_tensor, target


class DataCollatorReasonLangModeling:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    @jaxtyped(typechecker=typechecker)
    def collate_batch(self, batch: list[DatasetOutput]) -> BatchDataPoint:
        max_length = max(len(item) for item, _ in batch)
        padded_batch = []

        for item, _ in batch:
            num_padding = max_length - len(item)

            padded_sequence = item + [self.pad_token_id] * num_padding
            padded_batch.append(padded_sequence)

        padded_batch_tensor = torch.tensor(padded_batch)

        B, T = padded_batch_tensor.shape

        target = -100 * torch.ones(
            (B, T - 1), dtype=torch.long, device=padded_batch_tensor.device
        )

        # note: the following could probably be vectorizer, but at dataloading time
        # we don't care yet
        for b, (_, d_data) in enumerate(batch):
            cutoff = d_data["cutoff"]
            reason_net_token_num = d_data["reason_net_token_num"]

            assert cutoff > 0, "cutoff should be greater than 0"

            # everything before the end of the reason token is ignore by loss
            # until the last reason token (not included)
            target[b, 0 : cutoff + reason_net_token_num] = self.pad_token_id

            # target for everything after (including the last reason token)
            # the reason token is treated normally
            # aka target is the next token
            rest_roken = cutoff + reason_net_token_num
            target[b, rest_roken:] = padded_batch_tensor[b, rest_roken + 1 :]

        assert (target != -100).all(), "target should not contain -100 anymore"

        return padded_batch_tensor, target


class MathDataModule(L.LightningDataModule):
    train_prop: ClassVar[float] = 0.8

    def __init__(self, conf: MathDataConfig):
        super().__init__()
        self.conf = conf
        self.tokenizer = MathTokenizer()

    def prepare_data(self) -> None:
        self.data_collator: typing.Any

        if not self.conf.reason_net_data:
            self.dataset = MathDataset(self.conf.dataset_path, self.tokenizer)
            self.data_collator = DataCollatorLangModeling(self.tokenizer.pad_token_id)

        else:
            self.dataset = MathDatasetReason(
                self.conf.dataset_path,
                self.tokenizer,
                self.conf.reason_net_token_num,
            )
            self.data_collator = DataCollatorReasonLangModeling(
                self.tokenizer.pad_token_id
            )

    def setup(self, stage: str) -> None:
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

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
