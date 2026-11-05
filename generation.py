import argparse
import math
from pathlib import Path
import shlex
from jaxtyping import Bool, Float, Int
from torch import nn
import torch

from checkpointing import load_checkpoint
from config import parse_config
from cs336_basics.tokenizer import Tokenizer
import model as model_module
from model import softmax
import train


def generate(
    model: nn.Module,
    prompt_ids: Int[torch.Tensor, "... prompt"],
    max_tokens: int,
    temperature: float = 1.0,
    stop_tokens: list[int] = [],
    context_length: int = int(1e7),
    top_p = 1.0,
) -> Int[torch.Tensor, "... completion"]:
    assert max_tokens > 0
    assert temperature > 0
    completion_ids = []
    while len(completion_ids) < max_tokens:
        prompt = torch.cat([prompt_ids, *completion_ids], dim=-1)
        prompt = prompt[..., -context_length:]
        with torch.inference_mode():
            logits: Float[torch.Tensor, "... seq vocab"] = \
                model(prompt, mask=None, pos_ids=None)
        logits: Float[torch.Tensor, "... vocab"] = \
            logits[..., -1, :]
        logits = logits / temperature
        ps: Float[torch.Tensor, "... vocab"] = \
            softmax(logits, dim=-1)
        if top_p < 1.0:
            perm: Int[torch.Tensor, "... vocab"] = \
                ps.argsort(dim=-1, descending=True)
            ranks = perm.argsort(dim=-1, descending=False)
            sorted_ps = torch.gather(ps, dim=-1, index=perm)
            sorted_cum_ps = sorted_ps.cumsum(dim=-1)
            ks = torch.argmax(
                (sorted_cum_ps >= top_p).float(),
                dim=-1,
                keepdim=True
            )
            masked_ps = ps.masked_fill(ranks > ks, 0)
            assert torch.all(masked_ps.sum(dim=-1) >= top_p, dim=None)
            ps = masked_ps / masked_ps.sum(
                dim=-1, keepdim=True)
        completion_id: Int[torch.Tensor, "..."] = \
            torch.multinomial(ps, num_samples=1)
        completion_ids.append(completion_id)
        if any(
            id in stop_tokens
            for ids in completion_id.cpu().tolist()
            for id in ids
        ):
            break
    return torch.cat(completion_ids, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt')
    parser.add_argument('--max-tokens', default=32, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top-p', default=1.0, type=float)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32')

    parser.add_argument('--rundir', required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "encoding",
    )

    args = parser.parse_args()
    args.dtype = None if args.dtype is None else getattr(torch, args.dtype)
    args.rundir = Path(args.rundir).resolve()

    train_parser = train.get_argparser()
    train_args = parse_config(
        train_parser,
        ["--config", str(args.rundir / "config.yaml")],
    )
    if train_args.d_ff is None:  # Legacy
        train_args.d_ff = 64 * math.ceil(
            (8 * train_args.d_model) / (3 * 64)
        )


    model = model_module.TransformerLM(
        num_layers=train_args.num_layers,
        vocab_size=train_args.vocab_size,
        d_model=train_args.d_model,
        num_heads=train_args.num_heads,
        theta=train_args.theta,
        max_seq_len=train_args.context_length,
        d_ff=train_args.d_ff,
        device=args.device,
        dtype=args.dtype
    )
    cpt_file = args.rundir / "state.json"
    assert cpt_file.exists()
    _ = load_checkpoint(cpt_file, model, optim=None)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.data_dir / f"{train_args.dataset}-train-vocab.pk",
        merges_filepath=args.data_dir / f"{train_args.dataset}-train-merges.pk",
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    )
    prompt_ids = torch.tensor(tokenizer.encode(
        args.prompt)).unsqueeze(dim=0).to(args.device)
    completion_ids = generate(
        model,
        prompt_ids,
        args.max_tokens,
        args.temperature,
        tokenizer.encode('<|endoftext|>'),
        train_args.context_length,
        args.top_p,
    )
    completion = tokenizer.decode(
        completion_ids.cpu().squeeze(dim=0).tolist())
    print(completion)


if __name__ == "__main__":
    main()
