from typing import TypeAlias, TypeVar
from einops import rearrange

from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
import torch
import torch.nn.functional as F
from lightning import LightningModule
from beartype import beartype as typechecker
from reason_net.pydantic_conf import Config

from reason_net.data import BatchDataPoint, MathTokenizer
from reason_net.llama import LLaMA, LLaMaConfig

seq = TypeVar("seq")
b = TypeVar("b")


Batch: TypeAlias = tuple[Int[Tensor, "b seq_input"], Int[Tensor, "b seq_output"]]


class ModuleConfig(Config):
    model: LLaMaConfig
    lr: float


class LLaMaModule(LightningModule):
    def __init__(self, conf: ModuleConfig, tokenizer: MathTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        conf.model.vocab_size = tokenizer.vocab_size
        self.model = LLaMA(conf.model)
        self.conf = conf

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Int[Tensor, "b seq"]) -> Float[Tensor, "b seq vocab_size"]:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.conf.lr)

    @jaxtyped(typechecker=typechecker)
    def _loss_step(
        self, step_name: str, batch: BatchDataPoint, _batch_idx, accuracy: bool
    ) -> Float[Tensor, ""]:
        data, dict_data = batch
        input = data[:, :-1]

        if "target" in dict_data.keys():
            target = dict_data["target"]
        else:
            target = data[:, 1:]

        logits = self.forward(input)

        flatten_logits = rearrange(logits, "b seq vocab -> (b seq) vocab")
        flatten_target = rearrange(target, "b seq -> (b seq)")

        loss = F.cross_entropy(
            flatten_logits, flatten_target, ignore_index=self.tokenizer.pad_token_id
        )
        self.log(f"{step_name}_loss", loss)

        if accuracy:
            ignore_mask = target != self.tokenizer.pad_token_id
            token_acc = (
                (logits.argmax(dim=-1)[ignore_mask] == target[ignore_mask])
                .float()
                .mean()
            )
            self.log(f"{step_name}_token_accuracy", token_acc)

        return loss

    def training_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("train", batch, _batch_idx, accuracy=False)

    def validation_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("val", batch, _batch_idx, accuracy=True)
