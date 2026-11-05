from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parent

app = modal.App("cs336-smoke")
data_volume = modal.Volume.from_name("cs336-data")
runs_volume = modal.Volume.from_name("cs336-runs")

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
    "train.py",
    "generation.py",
    "config.py",
    "model.py",
    "data.py",
    "losses.py",
    "optimizers.py",
    "schedules.py",
    "checkpointing.py",
):
    image = image.add_local_file(
        ROOT / filename,
        f"/workspace/{filename}",
    )

image = (
    image
    .add_local_dir(
        ROOT / "cs336_basics",
        "/workspace/cs336_basics",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(ROOT / "configs", "/workspace/configs")
)


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={
        "/data": data_volume,
        "/runs": runs_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def pilot(
    run_name: str,
    config_name: str,
    mode: str,
    prompt: str,
):
    import time
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the remote environment.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"Python: {sys.executable}")

    rundir = Path("/runs") / run_name
    rundir.mkdir(parents=True, exist_ok=True)

    if mode == "train":
        command = [
            sys.executable, "-u", "/workspace/train.py",
            "--config", f"/workspace/configs/{config_name}",
            "--device", "cuda",
            "--data-dir", "/data/encoding",
            "--rundir", str(rundir),
        ]
    else:
        command = [
            sys.executable, "-u", "/workspace/generation.py",
            "--device", "cuda",
            "--data-dir", "/data/encoding",
            "--rundir", str(rundir),
            "--prompt", prompt,
            "--max-tokens", "64",
            "--temperature", "0.8",
        ]

    log_path = rundir / f"{mode}-{time.time_ns()}.log"

    try:
        with log_path.open("w") as log:
            with subprocess.Popen(
                command,
                cwd="/workspace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()

                returncode = process.wait()

            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
    finally:
        runs_volume.commit()


@app.local_entrypoint()
def main(
    run_name: str = "tinystories-modal-smoke",
    config_name: str = "smoke.yaml",
    mode: str = "train",
    prompt: str = "Once upon a time",
):
    if mode not in {"train", "generate"}:
        raise ValueError("mode must be train or generate")

    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run-name must be a single directory name")

    if Path(config_name).name != config_name:
        raise ValueError("config-name must be a filename within configs/")

    pilot.remote(run_name, config_name, mode, prompt)
