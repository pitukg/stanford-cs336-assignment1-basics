from jaxtyping import Int
import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    x: Int[npt.NDArray, "seq"],
    batch_size: int,
    context_length: int,
    device: torch.device,
    vocab_size: int | None = None
) -> tuple[
    Int[torch.LongTensor, "batch_size context_length"],
    Int[torch.LongTensor, "batch_size context_length"]
]:
    start_ids = np.random.choice(
        len(x) - context_length, size=batch_size, replace=True)
    input_ids = start_ids[:, None] + np.arange(context_length)[None, :]
    assert input_ids.shape == (batch_size, context_length)
    label_ids = 1 + input_ids
    inputs = torch.from_numpy(
        x[input_ids]
    ).to(device=device, dtype=torch.long)
    labels = torch.from_numpy(
        x[label_ids]
    ).to(device=device, dtype=torch.long)
    assert torch.all(0 <= inputs).item() and \
        torch.all(0 <= labels).item()
    if vocab_size is not None:
        # Sanity check
        assert torch.all(inputs < vocab_size).item() and \
            torch.all(labels < vocab_size).item()
    return (inputs, labels)
