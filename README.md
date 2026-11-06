# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## LR sweep

[Download the LR sweep charts (PDF)](reports/lr-sweep/lr-sweep.pdf)

![Training and validation learning curves](reports/lr-sweep/lr-sweep-preview.png)

The report contains training/validation cross-entropy, pre-clipping gradient norms,
interval clipping fractions, learning-rate schedules, and validation curves against
active-session time. The interrupted `1e-3` run combines both W&B sessions by
optimizer update; replayed updates use the newer session. Wall-clock curves sum
active-session timers, exclude the interruption gap, and estimate validation
timestamps from adjacent timing samples. See the PDF for measurement details.

Baseline: 4 layers, width 512, FFN 1344, 16 heads, context 256, vocabulary 10,000;
batch 128, 10,000 updates (327.68M tokens), float32 on A10. All trials use 1,000
warmup updates and cosine decay through update 10,000, with minimum LR equal to
one quarter of peak LR. AdamW uses betas `(0.9, 0.999)`, weight decay `0.1`, and
global gradient clipping at L2 norm `1.0`.

| Peak LR | Minimum LR | Final train CE (one batch) | Final validation CE |
| --- | --- | --- | --- |
| 3e-4 | 7.5e-5 | 1.5052 | 1.5124 |
| 1e-3 | 2.5e-4 | 1.3722 | 1.3834 |
| 3e-3 | 7.5e-4 | 1.3314 | 1.3540 |
| 1e-2 | 2.5e-3 | 1.3913 | 1.4300 |

The `3e-2` and `1e-1` curves are partial snapshots of ongoing runs, not final
results. The completed trials favor `3e-3` at this token budget; these are single
trials rather than multi-seed estimates. No causal attribution of gradient-norm
differences to weight decay is established by this sweep.

The exported [metric data](reports/lr-sweep/metrics.csv) and
[source run IDs, states, and export time](reports/lr-sweep/sources.json) accompany
the PDF. Live results are in the [W&B project](https://wandb.ai/pgab/cs336-basics).

