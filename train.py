import argparse
from functools import partial
import math
from pathlib import Path
import shlex
import sys
import numpy as np
from typing import Callable, TypeAlias

from jaxtyping import Float, Int
from torch import nn, optim
import torch

from checkpointing import load_checkpoint, save_checkpoint
from data import get_batch
from losses import cross_entropy
import model as model_module
import optimizers
from schedules import cosine_schedule


Loader: TypeAlias = Callable[[], tuple[
    Int[torch.LongTensor, "batch_size context_length"],
    Int[torch.LongTensor, "batch_size context_length"]
]]

def evaluate(
    model: nn.Module,
    loader: Loader,
    num_batches: int,
    device: torch.device | None
) -> None:
    print('==== Evaluating ====')
    num_tokens = []
    mean_losses = []
    for _ in range(num_batches):
        input_ids, labels = loader()
        with torch.inference_mode():
            logits: Float[torch.Tensor["... seq vocab"]] = \
                model(
                    input_ids.to(device),
                    mask=None,
                    pos_ids=None
                )

        losses: Float[torch.Tensor["... seq"]] = \
            cross_entropy(logits, labels.to(device))
        mean_losses.append(losses.mean(dim=-1).mean(dim=-1))  # Mean over batch dim too
        num_tokens.append(losses.shape[-1])
    weights = torch.tensor(num_tokens, dtype=torch.float)
    weights /= weights.sum()
    mean_loss = torch.dot(weights, torch.tensor(mean_losses))
    perplexity = torch.exp(mean_loss).cpu().item()
    print(f"{perplexity=:.4f}")


def train(
    model: nn.Module,
    train_loader: Loader,
    valid_loader: Loader,
    optim: optim.Optimizer,
    lr_schedule: Callable[int, float],
    max_grad_l2_norm: float,
    n_steps: int,
    device: torch.device | None,
    rundir: Path,
    eval_every_n_steps: int = 10,
    cpt_every_n_steps: int = 10,
) -> None:
    iter = 0
    if (cpt_file := rundir / "state.json").exists():
        print(f"Checkpoint found {cpt_file.absolute}, resuming...")
        iter = load_checkpoint(cpt_file, model, optim)
    print("Starting training")

    while iter < n_steps:
        iter += 1
        print(f"Iter {iter}/{n_steps}")
        input_ids, labels = train_loader()
        optim.zero_grad()
        logits: Float[torch.Tensor["... seq vocab"]] = \
            model(
                input_ids.to(device),
                mask=None,
                pos_ids=None
            )
        loss = cross_entropy(logits, labels.to(device)).mean(dim=None)
        print(f'{iter=}\tperplexity={torch.exp(loss.detach().cpu()).item():.2f}')
        loss.backward()
        optimizers.clip_grads(
            list(model.parameters()),
            max_grad_l2_norm
        )
        for group in optim.param_groups:
            group["lr"] = lr_schedule(iter)
        optim.step()

        if (iter-1) % cpt_every_n_steps == 0 or iter == n_steps:
            save_checkpoint(model, optim, iter, cpt_file)

        if (iter-1) % eval_every_n_steps == 0:
            evaluate(model, valid_loader, 16, device)

    # Final evaluation
    evaluate(model, valid_loader, 16, device)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-layers', required=True, type=int)
    parser.add_argument('--vocab-size', default=10_000, type=int)
    parser.add_argument('--d-model', required=True, type=int)
    parser.add_argument('--num-heads', required=True, type=int)
    parser.add_argument('--theta', default=10_000, type=float)
    parser.add_argument('--context-length', default=256, type=int)

    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32')

    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--weight-decay', required=True, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--max-grad-l2-norm', default=1.0, type=float)

    parser.add_argument('--dataset', default='TinyStoriesV2-GPT4')
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--n-steps', default=1, type=int)

    parser.add_argument('--rundir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()
    args.dtype = None if args.dtype is None else getattr(torch, args.dtype)
    args.rundir = Path(args.rundir).resolve()
    args.rundir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for file in args.rundir.iterdir():
            file.unlink()
    if not Path.exists(args.rundir / 'command.sh'):
        with open(args.rundir / 'command.sh', 'w') as f:
            f.write(shlex.join(sys.orig_argv))

    model = model_module.TransformerLM(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        theta=args.theta,
        max_seq_len=args.context_length,
        d_ff=64 * math.ceil((8 * args.d_model) / (3 * 64)),
        device=args.device,
        dtype=args.dtype
    )

    optim = optimizers.AdamW(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )

    train_filename, valid_filename = tuple(
        Path(__file__).parent /
            f'data/encoding/{args.dataset}'
            f'-{split}-tokenized.np'
        for split in ('train', 'valid')
    )
    train_dataset, valid_dataset = tuple(
        np.memmap(filename, dtype='<u2', mode='r')
        for filename in (train_filename, valid_filename)
    )
    # Demonstrate validation: don't do it here not to pull in everything into memory
    # for dataset in (train_dataset, valid_dataset):
    #     assert np.all(0 <= dataset) and \
    #         np.all(dataset < args.vocab_size)
    # Initial eval
    train_loader, valid_loader = tuple(
        partial(
            get_batch,
            x=x,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
            vocab_size=args.vocab_size,
        )
        for x in (train_dataset, valid_dataset)
    )
    lr_schedule = partial(
        cosine_schedule,
        lr_max=args.lr,
        lr_min=args.lr / 2,
        warmup=(args.n_steps * 1) // 4,
        final=(args.n_steps * 3) // 4,
    )

    train(model, train_loader, valid_loader, optim,
          lr_schedule, args.max_grad_l2_norm,
          args.n_steps, args.device, args.rundir)


if __name__ == "__main__":
    main()
