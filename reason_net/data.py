from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
import os
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias, TypeVar, TypedDict, TYPE_CHECKING
import typing
from einops import repeat

import numpy as np

from reason_net.pydantic_conf import Config
import torch
from torch import Tensor
import lightning as L
from torch.utils.data import Dataset, DataLoader
from jaxtyping import Int, jaxtyped
from beartype import beartype as typechecker

if TYPE_CHECKING:
    from reason_net.llama import MaskCache


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


class ReasonConfig(Config):
    reason_token_num: int = 20
    reason_token_pos: Literal["left", "middle"] = "middle"


class MathDataConfig(Config):
    seed: int = 42
    batch_size: int
    num_workers: int
    dataset_path: Path

    reason: ReasonConfig | None = None


seq = TypeVar("seq")
b = TypeVar("b")


DatasetOutput: TypeAlias = tuple[list[int], dict[str, int]]


class BaseMathDataset(Dataset, ABC):
    """
    This just load a file split by the equal sign and a define simple
    padded data collator
    """

    def __init__(
        self,
        dataset_path: Path,
        tokenizer: MathTokenizer,
        val: bool = False,
    ) -> None:
        super().__init__()

        if val:
            self.dataset_path = dataset_path / "val"
        else:
            self.dataset_path = dataset_path / "train"

        self.tokenizer = tokenizer

        self.chunks_files = os.listdir(self.dataset_path)

        if len(self.chunks_files) == 0:
            raise ValueError(f"No file in {self.dataset_path}")

        self.chunks_files.sort()

        data = []
        for chunk_file in self.chunks_files:
            with open(self.dataset_path / chunk_file, "r") as f:
                data.extend([line.strip() for line in f])

        self.data = np.array(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> DatasetOutput:
        return self.preprocess_data_point(self.data[idx])

    def split_data_point(self, data_point: str) -> tuple[list[int], list[int]]:
        [left, right] = data_point.split("=")

        data_left = self.tokenizer.encode(left)
        data_right = self.tokenizer.encode(right)

        return data_left, data_right

    @abstractmethod
    def preprocess_data_point(self, data_point: str) -> DatasetOutput:
        ...

    @jaxtyped(typechecker=typechecker)
    def collate_batch(self, batch: list[DatasetOutput]) -> BatchDataPoint:
        """
        This data collactor just pad the data to the left and mask
        the target before the equal sign
        """
        max_length = max(len(item) for item, _ in batch)

        padded_batch = []

        dict_data = defaultdict(list)

        for item, dict_item in batch:
            num_padding = max_length - len(item)

            padded_sequence = item + [self.tokenizer.pad_token_id] * num_padding
            padded_batch.append(padded_sequence)

            for key in dict_item.keys():
                dict_data[key].append(dict_item[key])

        padded_batch_tensor = torch.tensor(padded_batch)

        target = padded_batch_tensor[:, 1:].clone()

        for b, (_, d_data) in enumerate(batch):
            cutoff = d_data["cutoff"]
            target[b, 0:cutoff] = self.tokenizer.pad_token_id

        return {"data": padded_batch_tensor, "target": target, "attn_mask": None}


class MathDataset(BaseMathDataset):
    def preprocess_data_point(self, data_point: str) -> DatasetOutput:
        data_left, data_right = self.split_data_point(data_point)

        data = (
            data_left
            + [self.tokenizer.equal_token_id]  # add back the equal token
            + data_right
            + [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        )

        return data, {"cutoff": len(data_left)}


@jaxtyped(typechecker=typechecker)
def build_mask_cache() -> "MaskCache":
    ones = torch.ones(
        (64, 64),  # todo block size is hardcoded here
        dtype=torch.bool,
    )
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class MathDatasetReasonMiddle(BaseMathDataset):
    """
    add reason token at the left of the input
    Reason token appear like pad token in the target so that the
    model does not have to predict them
    """

    def __init__(
        self,
        dataset_path: Path,
        tokenizer: MathTokenizer,
        reason_token_num: int,
        val: bool = False,
    ) -> None:
        super().__init__(dataset_path, tokenizer, val=val)
        self.reason_token_num = reason_token_num

    def preprocess_data_point(self, data_point: str) -> DatasetOutput:
        data_left, data_right = self.split_data_point(data_point)

        data = (
            data_left
            + [self.tokenizer.equal_token_id]  # add back the equal token
            + [self.tokenizer.reason_token_id] * self.reason_token_num
            + data_right
            + [self.tokenizer.eos_token_id]
        )
        return data, {"cutoff": len(data_left)}

    @jaxtyped(typechecker=typechecker)
    def collate_batch(self, batch: list[DatasetOutput]) -> BatchDataPoint:
        tensor_batch = super().collate_batch(batch)

        mask = build_mask_cache()

        tensor_mask = repeat(
            mask,
            "l1 l2 s1 s2 -> b l1 l2 s1 s2",
            b=len(batch),
        )

        for mask, data_point in zip(tensor_mask, batch):
            cutoff = data_point[1]["cutoff"]
            mask[
                :,
                :,
                cutoff : cutoff + self.reason_token_num,
                cutoff : cutoff + self.reason_token_num,
            ] = True

        tensor_batch["attn_mask"] = tensor_mask

        return tensor_batch


class MathDatasetReasonLeft(BaseMathDataset):
    """
    add reason token at the left of the input.
    Reason token appear like pad token in the target so that the
    model does not have to predict them
    """

    def __init__(
        self,
        dataset_path: Path,
        tokenizer: MathTokenizer,
        reason_token_num: int,
        val: bool = False,
    ) -> None:
        super().__init__(dataset_path, tokenizer, val=val)
        self.reason_token_num = reason_token_num

    def preprocess_data_point(self, data_point: str) -> DatasetOutput:
        data_left, data_right = self.split_data_point(data_point)
        data = (
            [self.tokenizer.reason_token_id] * self.reason_token_num
            + data_left
            + [self.tokenizer.equal_token_id]  # add back the equal token
            + data_right
            + [self.tokenizer.eos_token_id]
        )
        return data, {
            "cutoff": len(data_left) + self.reason_token_num,
        }


class BatchDataPoint(TypedDict):
    data: Int[Tensor, "b seq"]
    target: Int[Tensor, "b seq_minus_one"]
    attn_mask: "MaskCache" | None


class MathDataModule(L.LightningDataModule):
    train_prop: ClassVar[float] = 0.8

    def __init__(self, conf: MathDataConfig):
        super().__init__()
        self.conf = conf
        self.tokenizer = MathTokenizer()

    def setup(self, stage: str) -> None:
        self.data_collator: typing.Any

        self.train: BaseMathDataset
        self.val: BaseMathDataset

        if not self.conf.reason:
            self.train = MathDataset(self.conf.dataset_path, self.tokenizer)
            self.val = MathDataset(self.conf.dataset_path, self.tokenizer, val=True)

        else:
            if self.conf.reason.reason_token_pos == "middle":
                self.train = MathDatasetReasonMiddle(
                    self.conf.dataset_path,
                    self.tokenizer,
                    reason_token_num=self.conf.reason.reason_token_num,
                )

                self.val = MathDatasetReasonMiddle(
                    self.conf.dataset_path,
                    self.tokenizer,
                    reason_token_num=self.conf.reason.reason_token_num,
                    val=True,
                )

            elif self.conf.reason.reason_token_pos == "left":
                self.train = MathDatasetReasonLeft(
                    self.conf.dataset_path,
                    self.tokenizer,
                    reason_token_num=self.conf.reason.reason_token_num,
                )
                self.val = MathDatasetReasonLeft(
                    self.conf.dataset_path,
                    self.tokenizer,
                    reason_token_num=self.conf.reason.reason_token_num,
                    val=True,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
            collate_fn=self.train.collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
            collate_fn=self.val.collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
