# credits to https://github.com/Lightning-AI/lit-llama/blob/main/generate.py

import sys
import time
from pathlib import Path
from typing import Literal, Optional, TypeAlias, TypeVar
from hydra import initialize, compose
import lightning as L
from omegaconf import OmegaConf
import torch
from cyclopts import App
from jaxtyping import Int

from reason_net.data.data import MathTokenizer
from reason_net.llama import Index, LLaMA, LLaMaConfig
from reason_net.module import LLaMaModule, ModuleConfig

app = App()

seq = TypeVar("seq")
InferenceIndex: TypeAlias = Int[torch.Tensor, "seq"]


@torch.no_grad()
def generate(
    model: LLaMA,
    idx: InferenceIndex,
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Index:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens

    assert isinstance(idx, InferenceIndex)  # type: ignore

    T = idx.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.conf.block_size)

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


def get_config(file_path: Path) -> LLaMaConfig:
    with initialize(
        version_base=None,
        config_path="../reason_net/configs/module/model",
    ):
        raw_conf = compose(config_name=str(file_path))

    OmegaConf.resolve(raw_conf)
    return LLaMaConfig(**OmegaConf.to_container(raw_conf))  # type: ignore


@app.default
def main(
    checkpoint_path: Path,
    model_conf: Literal["2M", "14M", "70M", "100k"],
    prompt: str,
    *,
    interactive: bool = False,
    num_samples: int = 1,
    max_new_tokens: int = 20,
    top_k: int = 200,
    temperature: float = 0.8,
    precision: Literal["bf16-true", "32-true"] = "32-true",
) -> None:
    """Generates text samples based on a pre-trained LLaMA model."""
    fabric = L.Fabric(devices=1, precision=precision)  # type: ignore

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    llama_conf = get_config(Path(f"{model_conf}.yaml"))
    module = LLaMaModule(ModuleConfig(model=llama_conf, lr=0.0))

    checkpoint = torch.load(checkpoint_path, map_location=fabric.device)
    module.load_state_dict(checkpoint["state_dict"])
    model = module.model
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)
    tokenizer = MathTokenizer()

    L.seed_everything(1234)

    continue_ = True

    while continue_:
        encoded = torch.Tensor(tokenizer.encode(prompt)).long().to(fabric.device)

        for i in range(num_samples):
            y = generate(
                model,
                encoded,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_id=tokenizer.eos_token_id,
            )

            model.reset_cache()
            print(tokenizer.decode(y.to("cpu").numpy().tolist()))

        if interactive:
            prompt = input("")
            if prompt == "":
                continue_ = False
        else:
            continue_ = False


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    app()
