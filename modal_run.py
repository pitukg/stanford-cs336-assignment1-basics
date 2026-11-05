import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import modal
import yaml


ROOT = Path(__file__).resolve().parent


def read_yaml(path):
    with path.open() as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def normalize(values):
    result = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError("Configuration keys must be strings")
        key = key.replace("-", "_")
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = value
    return result


def check_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Invalid run/trial name: {name!r}")


# This function executes remotely; it invokes your existing training script.
def run_trial(group, name, config, resume):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    runs = modal.Volume.from_name("cs336-runs")
    rundir = Path("/runs") / group / name
    manifest = rundir / "launch.yaml"
    checkpoint = rundir / "state.json"

    if resume:
        if not checkpoint.exists():
            raise RuntimeError(f"No checkpoint to resume: {checkpoint}")
        if not manifest.exists() or read_yaml(manifest) != config:
            raise RuntimeError("Resume configuration differs from original launch")
    else:
        # Refuse to silently resume or overwrite an existing experiment.
        rundir.mkdir(parents=True, exist_ok=False)
        with manifest.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=True)
        runs.commit()

    print(f"[{name}] GPU={torch.cuda.get_device_name(0)}")
    print(f"[{name}] Python={sys.executable}, torch={torch.__version__}")

    environment = os.environ.copy()
    environment["WANDB_RUN_GROUP"] = group

    command = [
        sys.executable, "-u", "/workspace/train.py",
        "--config", str(manifest),
    ]
    stamp = time.time_ns()
    started = time.perf_counter()
    result = {"trial": name, "status": "failed"}

    try:
        with (rundir / f"console-{stamp}.log").open("w") as log:
            with subprocess.Popen(
                command,
                cwd="/workspace",
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:
                for line in process.stdout:
                    print(f"[{name}] {line}", end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()

        result["returncode"] = returncode
        result["status"] = "completed" if returncode == 0 else "failed"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["elapsed_s"] = time.perf_counter() - started
        with (rundir / f"result-{stamp}.json").open("w") as f:
            json.dump(result, f, indent=2)
        runs.commit()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--trial", help="Run only this named trial")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sweep_path = args.sweep.resolve()
    spec = read_yaml(sweep_path)
    allowed = {"name", "base_config", "resources", "trials"}
    if set(spec) - allowed:
        raise ValueError(f"Unknown sweep keys: {set(spec) - allowed}")

    group = spec["name"]
    check_name(group)
    base_path = (sweep_path.parent / spec["base_config"]).resolve()
    base = normalize(read_yaml(base_path))

    resources = {
        "gpu": "A10",
        "cpu": 4,
        "memory_mb": 8192,
        "timeout_s": 14400,
        "concurrency": 1,
    }
    supplied_resources = spec.get("resources", {})
    if set(supplied_resources) - set(resources):
        raise ValueError("Unknown resource setting")
    resources.update(supplied_resources)
    for key in ("cpu", "memory_mb", "timeout_s", "concurrency"):
        if resources[key] <= 0:
            raise ValueError(f"{key} must be positive")

    # Validate keys against the actual training CLI.
    from train import get_argparser
    known = {action.dest for action in get_argparser()._actions}

    jobs = []
    names = set()
    for trial in spec["trials"]:
        if set(trial) - {"name", "overrides"}:
            raise ValueError(f"Unknown trial keys: {trial}")
        name = trial["name"]
        check_name(name)
        if name in names:
            raise ValueError(f"Duplicate trial: {name}")
        names.add(name)

        config = base | normalize(trial.get("overrides", {}))
        if set(config) - known:
            raise ValueError(f"Unknown training keys: {set(config) - known}")

        # Infrastructure settings always override the training YAML.
        config.update({
            "device": "cuda",
            "data_dir": "/data/encoding",
            "rundir": f"/runs/{group}/{name}",
            "overwrite": False,
        })
        if args.trial is None or args.trial == name:
            jobs.append((name, config))

    if not jobs:
        raise ValueError("No matching trials")

    print(yaml.safe_dump({
        "name": group,
        "resources": resources,
        "resume": args.resume,
        "trials": [{"name": name, "config": cfg} for name, cfg in jobs],
    }, sort_keys=False))
    print(f"{len(jobs)} trial(s); at most {resources['concurrency']} concurrent")

    if not args.execute:
        print("Dry run only. Add --execute to launch.")
        return

    image = (
        modal.Image.debian_slim(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        .uv_sync(
            uv_project_dir=str(ROOT),
            frozen=True,
            extra_options="--no-dev",
        )
        .env({"PYTHONUNBUFFERED": "1"})
    )
    for filename in (
        "train.py", "generation.py", "config.py", "model.py",
        "data.py", "losses.py", "optimizers.py",
        "schedules.py", "checkpointing.py",
    ):
        image = image.add_local_file(ROOT / filename, f"/workspace/{filename}")

    image = image.add_local_dir(
        ROOT / "cs336_basics",
        "/workspace/cs336_basics",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )

    app = modal.App("cs336-experiments")
    remote_trial = app.function(
        image=image,
        gpu=resources["gpu"],
        cpu=resources["cpu"],
        memory=resources["memory_mb"],
        timeout=resources["timeout_s"],
        max_containers=resources["concurrency"],
        volumes={
            "/data": modal.Volume.from_name("cs336-data"),
            "/runs": modal.Volume.from_name("cs336-runs"),
        },
        secrets=[modal.Secret.from_name("wandb-secret")],
    )(run_trial)

    results = []
    with modal.enable_output(), app.run():
        with ThreadPoolExecutor(max_workers=resources["concurrency"]) as pool:
            pending = {
                pool.submit(
                    remote_trial.remote, group, name, config, args.resume
                ): name
                for name, config in jobs
            }
            for future in as_completed(pending):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "trial": pending[future],
                        "status": "failed",
                        "error": str(exc),
                    }
                results.append(result)
                print(json.dumps(result), flush=True)

    print(json.dumps(results, indent=2))
    if any(result["status"] != "completed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
