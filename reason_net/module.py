from typing import TypeAlias, TypeVar

from jaxtyping import Int, Float, Bool, jaxtyped
from torch import Tensor
import torch
import torch.nn.functional as F
from lightning import LightningModule
from beartype import beartype as typechecker
from pydantic import BaseModel

from reason_net.data import BatchDataPoint
from reason_net.llama import LLaMA, LLaMaConfig

seq = TypeVar("seq")
b = TypeVar("b")


Batch: TypeAlias = tuple[Int[Tensor, "b seq_input"], Int[Tensor, "b seq_output"]]


class ModuleConfig(BaseModel):
    model: LLaMaConfig
    lr: float


class LLaMaModule(LightningModule):
    def __init__(self, conf: ModuleConfig):
        super().__init__()
        self.model = LLaMA(conf.model)
        self.conf = conf

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq vocab_size"]:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr)

    def _loss_step(
        self, step_name: str, batch: BatchDataPoint, _batch_idx
    ) -> Float[Tensor, ""]:
        data, start_end = batch

        output = self.forward(data)

        start_end_stack = torch.stack(start_end, dim=-1)
        mask = get_real_data_mask(start_end_stack, output.shape[1])

        output_for_loss = output[mask]
        target = data[mask]

        loss = F.cross_entropy(output_for_loss, target)
        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("train", batch, _batch_idx)

    def validation_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("val", batch, _batch_idx)


@jaxtyped(typechecker=typechecker)
def get_real_data_mask(
    all_start_end: Int[Tensor, "b 2"], seq: int
) -> Bool[Tensor, "b seq"]:
    b = all_start_end.shape[0]

    mask = torch.zeros(b, seq, dtype=torch.bool, device=all_start_end[0].device)

    for i, start_end in enumerate(all_start_end):
        start = start_end[0]
        end = start_end[1]
        try:
            mask[i, start:end] = True
        except IndexError:
            pass

    return mask
