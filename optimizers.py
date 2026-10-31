import math
from typing import Any, Callable, Iterable
from jaxtyping import Float
import torch
import torch.optim
from torch.optim.optimizer import ParamsT


class SGD(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr: float) -> None:
        super().__init__(params, defaults={
            "lr": lr
        })

    def step(self, closure: Callable[[], Any] | None = None) -> Any | None:
        loss: Any | None = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue  # <- why?
                state = self.state[p]
                t: int = state.get("t", 0)
                p.data -=  lr / math.sqrt(1 + t) * p.grad.data
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ) -> None:
        assert len(betas) == 2
        super().__init__(params, defaults={
            "lr": lr,
            "alpha": weight_decay,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": 1e-8
        })

    def step(self, closure: Callable[[], Any] | None = None) -> Any | None:
        loss: Any | None = None if closure is None else closure()
        for group in self.param_groups:
            lr, alpha = group["lr"], group["alpha"]
            beta1, beta2, eps = group["beta1"], group["beta2"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue  # <- why?
                state = self.state[p]
                t: int = state.get("t", 1)
                state["t"] = t + 1
                m: Float[torch.Tensor, "..."] | None = state.get("m", None)
                if m is None:
                    m = torch.zeros_like(p, dtype=p.dtype, device=p.device)
                v: Float[torch.Tensor, "..."] | None = state.get("v", None)
                if v is None:
                    v = torch.zeros_like(p, dtype=p.dtype, device=p.device)
                # Update first (m) and second (v) moments
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                state["m"] = m
                grad_sq = torch.square(p.grad.data)
                v = beta2 * v + (1 - beta2) * grad_sq
                state["v"] = v
                # Adjust estimators for startup bias
                m_adj = m / (1 - beta1**t)
                v_adj = v / (1 - beta2**t)
                # Weight decay
                p.data -= lr * alpha * p.data
                # Adaptive gradient step
                p.data -=  lr * m_adj / (torch.sqrt(v_adj) + eps)
        return loss


def clip_grads(
    params: Iterable[torch.nn.Parameter],
    max_grad_l2_norm: float,
    eps: float = 1e-6
) -> None:  # in place
    norm = torch.sqrt(sum(
        torch.norm(p.grad.data)**2
        for p in params
        if p.grad is not None
    ))
    if norm.cpu().item() <= max_grad_l2_norm:
        return
    else:
        scaler = max_grad_l2_norm / (norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data *= scaler


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    for lr in (1e1, 1e2, 1e3):
        print(f"{lr=}")
        opt = AdamW([weights], lr=lr, weight_decay=0.01)
        for t in range(100):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            if t % 10 == 0:
                print(f"t\t{loss.cpu().item():.5f}")
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
