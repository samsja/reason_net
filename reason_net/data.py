import random
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
    def __init__(self, conf: MathDataConfig):
        self.conf = conf

    def prepare_data(self) -> None:
        self.generator = MathDataGen(self.conf.min, self.conf.max)
        self.tokenizer = MathTokenizer(MathDataGen.operand)

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders

        if stage != "fit":
            raise NotImplementedError(f"DataModule stage {stage} not implemented")

        full_raw = [self.generator.generate() for _ in range(self.conf.size)]

        full_tokenized = [
            (self.tokenizer.encode(data), self.tokenizer.encode(target))
            for data, target in full_raw
        ]

        max_exo = max([len(exo) for exo, _ in full_tokenized])
        max_resp = max([len(resp) for _, resp in full_tokenized])

        for i, (exo, resp) in enumerate(full_tokenized):
            exo = exo + [self.tokenizer.pad_token_id] * (max_exo - len(exo))
            resp = resp + [self.tokenizer.pad_token_id] * (max_resp - len(resp))

            full_tokenized[i] = (torch.Tensor(exo).long(), torch.Tensor(resp).long())

        self.full_dataset = ListDataset(full_tokenized)

        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size

        self.train, self.val = random_split(
            self.full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.conf.seed),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.conf.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.conf.batch_size)
