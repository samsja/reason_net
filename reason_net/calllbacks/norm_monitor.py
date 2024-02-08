from __future__ import annotations

from functools import partial
from pydantic import PrivateAttr
import torch

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Callback

from typing import TYPE_CHECKING, Any, Type, no_type_check

from reason_net.pydantic_conf import Config

if TYPE_CHECKING:
    from reason_net.module import LLaMaModule
    from lightning.pytorch import Trainer


class NormMonitor(Callback):
    @no_type_check
    def __init__(self, config: NormMonitorConfig) -> None:
        super().__init__()
        self.config = config

        self.handles = []

    @no_type_check
    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: "Trainer",
        pl_module: "LLaMaModule",
        *args,
        **kwargs,
    ) -> None:
        """Register the hooks to monitor the activations."""
        if trainer.global_step % self.config.log_every_n_steps == 0:

            def _hook(
                name: str,
                _mod: Any,
                _inp: Any,
                outp: torch.Tensor,
            ) -> None:
                norm = outp.norm(p=2)
                pl_module.log(f"norm/{name}", norm)

            for i, layer in enumerate(pl_module.model.transformer.h):
                _h = layer.register_forward_hook(partial(_hook, f"layer_{i}"))
                self.handles.append(_h)

            _h = pl_module.model.lm_head.register_forward_hook(
                partial(_hook, "lm_head")
            )
            self.handles.append(_h)

    @no_type_check
    @rank_zero_only
    def on_train_batch_end(self, *args, **kwargs) -> None:
        """Remove the hooks after the batch ends."""
        for h in self.handles:
            h.remove()


class NormMonitorConfig(Config):
    log_every_n_steps: int
    _target_: Type[Callback] = PrivateAttr(NormMonitor)
