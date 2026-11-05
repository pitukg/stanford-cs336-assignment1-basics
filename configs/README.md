# Matched smoke runs

Run commands from the repository root. Both runs use `configs/smoke.yaml`:
2 layers, width 128, 4 heads, context 128, batch 8, float32, and 200 updates
(204,800 training tokens). Device and filesystem paths differ. Random seeds
are not fixed, so this matches hyperparameters, not exact random samples.

Local prerequisites: tokenized train/valid files and tokenizer artifacts in
`data/encoding`, dependencies installed with `uv sync`, and `uv run wandb login`.

```bash
uv run python train.py --config configs/smoke.yaml --device cpu --rundir .runs/smoke-local-01
```

Modal prerequisites: authenticated CLI, `cs336-data` Volume containing those
same artifacts under `/encoding`, `cs336-runs` Volume, and `wandb-secret`
containing `WANDB_API_KEY` in the active workspace/environment.
The launcher uses A10, native frozen uv sync, and persistent run outputs.

```bash
uv run modal run --detach modal_pilot.py --config-name smoke.yaml --run-name smoke-modal-01
```

Use fresh run names for fresh comparisons. Reusing a directory resumes its
checkpoint; a completed 200-step run will only evaluate on relaunch.
W&B creates one run per invocation, grouped by run-directory name.
Compare final validation metrics, not a single training batch against validation.

Generation checks:

```bash
uv run python generation.py --rundir .runs/smoke-local-01 --device cpu --prompt "Once upon a time" --max-tokens 64 --temperature 0.8
uv run modal run modal_pilot.py --run-name smoke-modal-01 --mode generate --prompt "Once upon a time"
```

Inspect and download remote outputs:

```bash
uv run modal volume ls cs336-runs /smoke-modal-01
uv run modal volume get cs336-runs /smoke-modal-01 .runs/smoke-modal-download-01
```

The recorded session timer excludes final checkpointing, final evaluation,
and W&B shutdown. Interval throughput includes intervening evaluation and
checkpoint overhead. Queue and image-build time are separate.

Verified pilot: `tinystories-modal-wandb-smoke`, W&B run
https://wandb.ai/pgab/cs336-basics/runs/si9r5f4u completed 200 updates with
final validation perplexity 91.2003; its checkpoint and logs persisted.
Generation and checkpoint-resume checks are separate from that verification.
