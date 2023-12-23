import random

import torch
import lightning as L
from torch.utils.data import Dataset, random_split


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

    def __init__(self, operand: list[str]):
        self.vocab = {str(i): i for i in range(self.max_digit)}
        for i, op in enumerate(operand):
            self.vocab[op] = self.max_digit + i
        self.anti_vocab = {value: key for key, value in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, x: str) -> list[int]:
        return list(map(lambda x: self.vocab[x], list(x)))

    def decode(self, x: list[int]) -> str:
        decoded = list(map(lambda x: self.anti_vocab[x], list(x)))
        return "".join(decoded)


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MathDataModule(L.LightningDataModule):
    def __init__(self, min: int, max: int, size: int, seed: int):
        self.size = size

        self.min = min
        self.max = max
        self.seed = seed

    def prepare_data(self):
        self.generator = MathDataGen(min, max)
        self.tokenizer = MathTokenizer(MathDataGen.operand)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        match stage:
            case "fit":
                full_raw = [self.generator.generate() for _ in range(self.size)]
                full_tokenized = [
                    (self.tokenizer.encode(data), self.tokenizer.encode(target))
                    for data, target in full_raw
                ]
                full_dataset = ListDataset(full_tokenized)

                train_size = int(0.8 * len(full_dataset))
                val_size = len(full_dataset) - train_size

                self.train, self.val = random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed),
                )

            case _:
                raise NotImplementedError()
