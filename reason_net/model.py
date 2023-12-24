from typing import Protocol, TypeAlias

from jaxtyping import Int, Float
from pydantic import BaseModel
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from torch.nested import nested_tensor
from lightning import LightningModule
from einops import rearrange

from reason_net.data import BatchDataPoint


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

        output_for_loss = nested_tensor(
            [output[:, cutoff:, :] for cutoff in all_cutoff]
        )
        target = nested_tensor([data[:, cutoff:] for cutoff in all_cutoff])

        output_for_loss = rearrange(output_for_loss, "b seq vocab -> (b seq) vocab")
        target = rearrange(target, "b seq -> (b seq)")

        loss = F.cross_entropy(output_for_loss, target)
        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("train", batch, _batch_idx)

    def validation_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("val", batch, _batch_idx)
