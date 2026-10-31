from jaxtyping import Float, Int
from torch import Tensor
import torch

from model import log_softmax


def cross_entropy(
    logits: Float[Tensor, "... vocab"],
    labels: Int[Tensor, "..."]
) -> Float[Tensor, "..."]:
    logp: Float[Tensor, "... vocab"] = log_softmax(logits, dim=-1)
    # gathered_logp: Float[Tensor, "..."] = logp[labels]  <- incorrect!
    gathered_logp: Float[Tensor, "..."] = torch.gather(
        logp, dim=-1, index=labels[..., None])
    return -gathered_logp.squeeze(dim=-1)


def perplexity(
    logits: Float[Tensor, "... seq vocab"],
    labels: Int[Tensor, "... seq"]
) -> Float[Tensor, "..."]:
    losses: Float[Tensor["... seq"]] = cross_entropy(logits, labels)
    perp = torch.exp(losses.mean(dim=-1))
    return perp
