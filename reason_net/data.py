import random

# import torch
# from torch.utils.data import DataLoader, Dataset


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
