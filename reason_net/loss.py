from typing import TypeAlias, TypeVar
from einops import rearrange, repeat

from torch import Tensor, nn
import torch
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


def sim_clr_loss(x: Float[Tensor, "batch seq n_embd"]) -> SingleFloat:
    x = F.normalize(x, dim=-1)
    x_t = rearrange(x, "batch seq n_embd -> batch n_embd seq")
    sim = x @ x_t

    seq, b = sim.shape[1], sim.shape[0]

    eye = repeat(
        torch.eye(seq, device=sim.device, dtype=sim.dtype), "s1 s2-> b s1 s2", b=b
    )

    return torch.norm(sim - eye, p=2)
