from typing import Protocol
from jaxtyping import Int, Float
from torch import Tensor, nn


class GPTLike(Protocol):
    def forward(self, x: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq vocab_size"]:
        ...


class DummyGPT(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()

        self.we = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq vocab_size"]:
        x = self.we(x)
        return self.linear(x)
