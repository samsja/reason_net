from typing import Type, TypeAlias, TypeVar
from einops import rearrange

from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
import torch
import torch.nn.functional as F
from lightning import LightningModule
from beartype import beartype as typechecker
from pydantic import BaseModel

from reason_net.data import BatchDataPoint, MathTokenizer
from reason_net.llama import LLaMA, LLaMaConfig

seq = TypeVar("seq")
b = TypeVar("b")


Batch: TypeAlias = tuple[Int[Tensor, "b seq_input"], Int[Tensor, "b seq_output"]]


class ModuleConfig(BaseModel):
    model: LLaMaConfig
    lr: float


class NormalModule(LightningModule):
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
        data, _ = batch
        input = data[:, :-1]
        target = data[:, 1:]

        logits = self.forward(input)

        flatten_logits = rearrange(logits, "b seq vocab -> (b seq) vocab")
        flatten_target = rearrange(target, "b seq -> (b seq)")

        loss = F.cross_entropy(
            flatten_logits, flatten_target, ignore_index=-self.tokenizer.pad_token_id
        )
        self.log(f"{step_name}_loss", loss)

        if accuracy:
            token_acc = (logits.argmax(dim=-1) == target).float().mean()
            self.log(f"{step_name}_token_accuracy", token_acc)

        return loss

    def training_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("train", batch, _batch_idx, accuracy=False)

    def validation_step(self, batch: BatchDataPoint, _batch_idx) -> Float[Tensor, ""]:
        return self._loss_step("val", batch, _batch_idx, accuracy=True)


# todo challenge inheritance tree, use an abstract class instead
# reason net should not depend on NormalModule directly
class ReasonNetModule(NormalModule):
    @jaxtyped(typechecker=typechecker)
    def _loss_step(
        self, step_name: str, batch: BatchDataPoint, _batch_idx, accuracy: bool
    ) -> Float[Tensor, ""]:
        data, dict_data = batch

        B, T = data.shape

        input = data[:, :-1]

        target = -100 * torch.ones((B, T - 1), dtype=torch.long, device=data.device)

        for b in range(B):
            cutoff = dict_data["cutoff"][b]
            reason_token_num = dict_data["reason_token_num"][b]

            assert cutoff > 0, "cutoff should be greater than 0"

            # everything before the equal sign is treated normally
            # aka target is the next token
            target[b, 0:cutoff] = data[b, 1 : cutoff + 1]

            last_reason_token_pos = cutoff + reason_token_num
            first_reason_token_pos = cutoff + 1
            # target for the equal token is the token after the last reason token
            target[b, cutoff] = data[b, last_reason_token_pos + 1]

            # target for the reason token is the pad token, because we want to ignore it
            target[
                b, first_reason_token_pos : last_reason_token_pos + 1
            ] = self.tokenizer.pad_token_id

            # target for everything after the reason token is treated normally
            # aka target is the next token
            rest_roken = last_reason_token_pos + 1
            target[b, rest_roken:] = data[b, rest_roken + 1 :]

        assert (target != -100).all(), "target should not contain -100 anymore"

        logits = self.forward(input)

        flatten_logits = rearrange(logits, "b seq vocab -> (b seq) vocab")
        flatten_target = rearrange(target, "b seq -> (b seq)")

        loss = F.cross_entropy(
            flatten_logits, flatten_target, ignore_index=self.tokenizer.pad_token_id
        )
        self.log(f"{step_name}_loss", loss)

        if accuracy:
            token_acc = (logits.argmax(dim=-1) == target).float().mean()
            self.log(f"{step_name}_token_accuracy", token_acc)

        return loss


NAME_TO_MODULE: dict[str, Type[NormalModule]] = {
    "normal": NormalModule,
    "reason": ReasonNetModule,
}
