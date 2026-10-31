import argparse
import math
from pathlib import Path
import shlex
from jaxtyping import Float, Int
from torch import nn
import torch

from checkpointing import load_checkpoint
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
        # TODO max p if desired
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
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32')

    parser.add_argument('--rundir', required=True)

    args = parser.parse_args()
    args.dtype = None if args.dtype is None else getattr(torch, args.dtype)
    args.rundir = Path(args.rundir).resolve()

    train_parser = train.get_argparser()
    with open(args.rundir / 'command.sh', 'r') as f:
        saved_command = f.read()
        saved_args = shlex.split(saved_command)
        prefix_len = 3 if saved_args[1] == "-m" else 2
        train_args = train_parser.parse_args(
            saved_args[prefix_len:])

    model = model_module.TransformerLM(
        num_layers=train_args.num_layers,
        vocab_size=train_args.vocab_size,
        d_model=train_args.d_model,
        num_heads=train_args.num_heads,
        theta=train_args.theta,
        max_seq_len=train_args.context_length,
        d_ff=64 * math.ceil((8 * train_args.d_model) / (3 * 64)),
        device=args.device,
        dtype=args.dtype
    )
    cpt_file = args.rundir / "state.json"
    assert cpt_file.exists()
    _ = load_checkpoint(cpt_file, model, optim=None)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=Path(__file__).parent / \
            "data/encoding/TinyStoriesV2-GPT4-train-vocab.pk",
        merges_filepath=Path(__file__).parent / \
            "data/encoding/TinyStoriesV2-GPT4-train-merges.pk",
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
        train_args.context_length
    )
    completion = tokenizer.decode(
        completion_ids.cpu().squeeze(dim=0).tolist())
    print(completion)


if __name__ == "__main__":
    main()
