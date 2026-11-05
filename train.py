import argparse
from functools import partial
import itertools
import math
from pathlib import Path
import shlex
import sys
import time
import numpy as np
from typing import Callable, TypeAlias

from jaxtyping import Float, Int
from torch import nn, optim
import torch
import wandb

from checkpointing import load_checkpoint, save_checkpoint
from config import parse_config, save_resolved_config
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
    device: torch.device | None,
    step: int | None = None,
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
    if wandb.run is not None and step is not None:
        wandb.log({
            "trainer/step": step,
            "val/cross_entropy": mean_loss.item(),
            "val/perplexity": perplexity,
        })


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
    eval_every_n_steps: int = 50,
    cpt_every_n_steps: int = 50,
    log_every_n_steps: int = 5,
) -> None:
    iter, cum_tokens_processed = 0, 0
    if (cpt_file := rundir / "state.json").exists():
        print(f"Checkpoint found {cpt_file.absolute}, resuming...")
        iter = load_checkpoint(cpt_file, model, optim)
    print("Starting training")
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    perf_counter, tokens_processed = time.perf_counter(), 0
    start_perf_counter = perf_counter

    while iter < n_steps:
        if iter % eval_every_n_steps == 0:
            evaluate(model, valid_loader, 16, device, iter)

        iter += 1
        print(f"Iter {iter}/{n_steps}")
        input_ids, labels = train_loader()
        cum_tokens_processed += torch.tensor(input_ids.shape).prod().item()
        optim.zero_grad()
        logits: Float[torch.Tensor["... seq vocab"]] = \
            model(
                input_ids.to(device),
                mask=None,
                pos_ids=None
            )
        loss = cross_entropy(logits, labels.to(device)).mean(dim=None)
        loss.backward()
        optimizers.clip_grads(
            list(model.parameters()),
            max_grad_l2_norm
        )
        lr = lr_schedule(iter)
        for group in optim.param_groups:
            group["lr"] = lr
        optim.step()

        if (iter-1) % log_every_n_steps == 0 or iter == n_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            elapsed = now - perf_counter
            total_elapsed = now - start_perf_counter
            tokens_elapsed = cum_tokens_processed - tokens_processed
            tokens_processed = cum_tokens_processed
            perf_counter = now
            print(
                f"step={iter}"
                f"\tsession_tokens={cum_tokens_processed / 1e6:,.3f}M"
                f"\tinterval={elapsed:.2f}s"
                f"\tsession_total={total_elapsed:.2f}s"
                f"\tthroughput={tokens_elapsed / elapsed / 1e3:,.2f}k tok/s"
                f"\tlr={lr:.2e}"
                f"\tperplexity={torch.exp(loss.detach().cpu()).item():.2f}"
            )
            if wandb.run is not None:
                loss_value = loss.detach().cpu().item()
                wandb.log({
                    "trainer/step": iter,
                    "train/cross_entropy": loss_value,
                    "train/perplexity": math.exp(loss_value),
                    "train/lr": lr,
                    "perf/session_tokens": cum_tokens_processed,
                    "perf/session_elapsed_s": total_elapsed,
                    "perf/interval_s": elapsed,
                    "perf/tokens_per_s": tokens_elapsed / elapsed,
                })

        if (iter-1) % cpt_every_n_steps == 0 or iter == n_steps:
            save_checkpoint(model, optim, iter, cpt_file)

    # Final evaluation
    evaluate(model, valid_loader, 16, device, iter)


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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "encoding",
    )
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--n-steps', default=1, type=int)

    parser.add_argument('--rundir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cs336-basics")
    parser.add_argument("--wandb-entity", default="")

    return parser


def main():
    parser = get_argparser()
    args = parse_config(parser)

    args.rundir = Path(args.rundir).resolve()
    args.rundir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for file in args.rundir.iterdir():
            file.unlink()

    # Save before converting the dtype string into a torch.dtype object.
    # Keep earlier invocation configurations when resuming.
    config_path = args.rundir / "config.yaml"
    if config_path.exists():
        config_path = args.rundir / f"config-{time.time_ns()}.yaml"
    save_resolved_config(args, config_path)

    args.dtype = getattr(torch, args.dtype)

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

    train_filename, valid_filename = (
        args.data_dir / f"{args.dataset}-{split}-tokenized.np"
        for split in ("train", "valid")
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

    if args.wandb:
        run_config = {
            key: str(value) if isinstance(value, (Path, torch.dtype)) else value
            for key, value in vars(args).items()
        }

        with wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=f"{args.rundir.name}-{time.time_ns()}",
            group=args.rundir.name,
            config=run_config,
            dir=str(args.rundir),
            job_type="train",
        ) as run:
            run.define_metric("trainer/step")
            run.define_metric("train/*", step_metric="trainer/step")
            run.define_metric("val/*", step_metric="trainer/step")
            run.define_metric("perf/*", step_metric="trainer/step")

            train(
                model, train_loader, valid_loader, optim,
                lr_schedule, args.max_grad_l2_norm,
                args.n_steps, args.device, args.rundir,
            )
    else:
        train(
            model, train_loader, valid_loader, optim,
            lr_schedule, args.max_grad_l2_norm,
            args.n_steps, args.device, args.rundir,
        )


if __name__ == "__main__":
    main()
