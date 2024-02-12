from typing import TypeAlias, TypeVar

from torch import Tensor, nn
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker

X = TypeVar("X")

Logits: TypeAlias = Float[Tensor, "X n_embd"]
Target: TypeAlias = Int[Tensor, "X"]

SingleFloat: TypeAlias = Float[Tensor, ""]


class MaxZLoss(nn.CrossEntropyLoss):
    """MaxZLoss.

    from the baichuan2 paper: https://arxiv.org/abs/2309.10305

    .. math::
        z_{loss} = weight z^{2}

    where z is the max logit
    """

    def __init__(self, z_loss_w: float, ignore_index: int) -> None:
        super().__init__(ignore_index=ignore_index)
        self.z_loss_w = z_loss_w

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, logits: Logits, target: Target
    ) -> tuple[SingleFloat, SingleFloat | None]:
        loss = super().forward(logits, target)

        max_logits = logits.max(dim=-1)[0]
        max_logits = max_logits.where(target != self.ignore_index, 0)
        # max is not differentiable. But here we just pick the indices of the max
        # value, so it's fine for backpropagation.

        z_loss = self.z_loss_w * max_logits.pow(2).mean()
        return loss, z_loss


class EmebddingEntropyMinimizer(nn.Module):
    """
    This loss is mean to increase the entropy of a list of embeddings.

    It is inspired by the Mean Entropy Maximization (ME-MAX) regularizer.

    In a nutshell it tries to maximize the entropy over a learn distribution.

    """

    def __init__(self, in_features: int, out_features) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: Logits) -> SingleFloat:
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1).mean()
