from typing import Protocol, TypeAlias, TypeVar

from jaxtyping import Int, Float, Bool
from pydantic import BaseModel
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from lightning import LightningModule

from reason_net.data import BatchDataPoint

seq = TypeVar("seq")
b = TypeVar("b")


class GPTLikeConfig(BaseModel):
    vocab_size: int
    hidden_dim: int


class GPTLike(Protocol):
    def __init__(self, conf: GPTLikeConfig):
        ...

    def forward(self, x: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq vocab_size"]:
        ...


class DummyGPTConfig(GPTLikeConfig):
    ...


class DummyGPT(nn.Module):
    def __init__(self, conf: DummyGPTConfig):
        super().__init__()
        self.conf = conf
        self.we = nn.Embedding(conf.vocab_size, conf.hidden_dim)
        self.linear = nn.Linear(conf.hidden_dim, conf.vocab_size)

    def forward(self, x: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq vocab_size"]:
        x = self.we(x)
        return self.linear(x)


class GPTModuleConfig(BaseModel):
    model: DummyGPTConfig
    lr: float


Batch: TypeAlias = tuple[Int[Tensor, "b seq_input"], Int[Tensor, "b seq_output"]]


class GPTModule(LightningModule):
    def __init__(self, conf: GPTModuleConfig):
        super().__init__()
        self.model = DummyGPT(conf.model)
        self.conf = conf

    def forward(self, x: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq vocab_size"]:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr)

    def _loss_step(
        self, step_name: str, batch: BatchDataPoint, _batch_idx
    ) -> Float[Tensor, ""]:
        data, all_cutoff = batch
        output = self.forward(data)

        cutoff_mask = get_cutoff_mask(all_cutoff, output.shape[1])

        output_for_loss = output[cutoff_mask]
        target = data[cutoff_mask]

        loss = F.cross_entropy(output_for_loss, target)
        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("train", batch, _batch_idx)

    def validation_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("val", batch, _batch_idx)


def get_cutoff_mask(all_cutoff: Int[Tensor, "b"], seq: int) -> Bool[Tensor, "b seq"]:
    b = all_cutoff.shape[0]

    mask = torch.zeros(b, seq, dtype=torch.bool, device=all_cutoff.device)

    for i, cutoff in enumerate(all_cutoff):
        mask[i, cutoff:] = True

    return mask
