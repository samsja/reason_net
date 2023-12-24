from typing import TypeAlias, TypeVar

from jaxtyping import Int, Float, Bool, jaxtyped
from torch import Tensor
import torch
import torch.nn.functional as F
from lightning import LightningModule
from beartype import beartype as typechecker
from pydantic import BaseModel

from reason_net.data import BatchDataPoint
from reason_net.llama import LLaMaConfig

seq = TypeVar("seq")
b = TypeVar("b")


Batch: TypeAlias = tuple[Int[Tensor, "b seq_input"], Int[Tensor, "b seq_output"]]


class ModuleConfig(BaseModel):
    model: LLaMaConfig
    lr: float


class LLamModule(LightningModule):
    def __init__(self, conf: ModuleConfig):
        super().__init__()
        self.model = LLaMaConfig(conf.model)
        self.conf = conf

    @jaxtyped(typechecker=typechecker)
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


@jaxtyped(typechecker=typechecker)
def get_cutoff_mask(all_cutoff: Int[Tensor, "b"], seq: int) -> Bool[Tensor, "b seq"]:
    b = all_cutoff.shape[0]

    mask = torch.zeros(b, seq, dtype=torch.bool, device=all_cutoff.device)

    for i, cutoff in enumerate(all_cutoff):
        mask[i, cutoff:] = True

    return mask
