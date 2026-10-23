from einops import rearrange, reduce
from jaxtyping import Bool, Float, Int
import math
from typing import Any
import torch
from torch import Tensor
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *args: Any,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.W = nn.Parameter(
            torch.empty((out_features, in_features),
                        dtype=dtype, device=device)
        )
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=sigma,
                                    a=-3*sigma, b=3*sigma)

    def forward(self, x: Float[Tensor, " ... d_in"]
               ) -> Float[Tensor, " ... d_out"]:
        return torch.einsum("oi,...i->...o", self.W, x)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *args,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.table = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim),
                        device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.table, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, "..."]
               ) -> Float[Tensor, "... d_model"]:
        return self.table[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *args,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_sq = reduce(torch.square(x), "... d_model -> ...", "mean")
        rms = torch.sqrt(mean_sq + self.eps)
        result = x / rms[..., None] * self.gain
        # Return the result in the original dtype
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *args,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.W1 = nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )
        self.W2 = nn.Parameter(
            torch.empty((d_model, d_ff), device=device, dtype=dtype)
        )
        self.W3 = nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        pre_gate = torch.einsum("fd,...d->...f", self.W1, x)
        gate = pre_gate * torch.sigmoid(pre_gate)
        up_proj = torch.einsum("fd,...d->...f", self.W3, x)
        out = torch.einsum("df,...f->...d", self.W2, up_proj * gate)
        return out


class RoPE(nn.Module):
    def __init__(
        self,
        d_kv: int,
        theta: float,
        max_seq_len: int,
        *args,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.d_kv = d_kv
        assert self.d_kv % 2 == 0, "Expecting even d_kv for RoPE"
        self.Theta = theta
        # Precompute sin/cos as a non-persistent buffer
        k: Int[Tensor, "k"] = 1 + torch.arange(
            self.d_kv // 2, device=device)
        all_pos_ids: Int[Tensor, "seq"] = torch.arange(
            max_seq_len, device=device)
        theta: Float[Tensor, "seq k"] = all_pos_ids[..., None] / \
            torch.pow(self.Theta, (2 * k - 2) / self.d_kv)
        if dtype is not None:
            theta = theta.to(dtype)
        self.sin: Float[Tensor, "seq k"] = nn.Buffer(
            torch.sin(theta), persistent=False)  # Don't keep in state dict
        self.cos: Float[Tensor, "seq k"] = nn.Buffer(
            torch.cos(theta), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq d_kv"],
        pos_ids: Int[Tensor, "... seq"],
    ) -> Float[Tensor, "... seq d_kv"]:
        sin: Float[Tensor, "... seq k"] = self.sin[pos_ids]
        cos: Float[Tensor, "... seq k"] = self.cos[pos_ids]

        x_reshaped : Float[Tensor, "... seq d_kv 2"] = rearrange(
            x, "... seq (d_kv_half two) -> ... seq d_kv_half two", two=2)
        result: Float[Tensor, "... seq d_kv_half 2"] = torch.empty_like(x_reshaped)
        result[..., 0] = cos * x_reshaped[..., 0] - sin * x_reshaped[..., 1]
        result[..., 1] = sin * x_reshaped[..., 0] + cos * x_reshaped[..., 1]

        result: Float[Tensor, "... seq d_kv"] = rearrange(
            result, "... seq d_kv_half two -> ... seq (d_kv_half two)")
        return result
