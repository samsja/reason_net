# credits to https://github.com/Lightning-AI/lit-llama/blob/main/generate.py

from pathlib import Path
from typing import TypeAlias, TypeVar
from hydra import initialize, compose
from omegaconf import OmegaConf
import torch
from cyclopts import App
from jaxtyping import Int

from reason_net.llama import LLaMA, LLaMaConfig

app = App()

seq = TypeVar("seq")
InferenceIndex: TypeAlias = Int[torch.Tensor, "seq"]


@torch.no_grad()
def generate(
    model: LLaMA,
    idx: InferenceIndex,
    max_new_tokens: int,
    *,
    eos_id: int | None = None,
) -> Int[torch.Tensor, "seq"]:
    device, dtype = idx.device, idx.dtype
    new_data = torch.empty(1, idx.size(0) + max_new_tokens, dtype=dtype, device=device)
    pos = len(idx)
    new_data[0, :pos] = idx

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        logits = model(new_data[:, :pos])
        logits = logits[0, -1]

        idx_next = logits.argmax(dim=-1).to(dtype=dtype)

        new_data[0, pos] = idx_next

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return new_data[0, :pos]
        pos += 1

    return new_data[0]


def get_config(file_path: Path) -> LLaMaConfig:
    with initialize(
        version_base=None,
        config_path="../reason_net/configs/module/model",
    ):
        raw_conf = compose(config_name=str(file_path))

    OmegaConf.resolve(raw_conf)
    return LLaMaConfig(**OmegaConf.to_container(raw_conf))  # type: ignore
