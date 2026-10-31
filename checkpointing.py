import os
import typing
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'iter': iteration,
    }, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optim: torch.optim.Optimizer | None = None
) -> int:
    state = torch.load(src)
    assert {'model', 'optim', 'iter'}.issubset(state.keys())
    model.load_state_dict(state['model'])
    if optim is not None:
        optim.load_state_dict(state['optim'])
    return state['iter']
