from einops import einsum, rearrange, reduce
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
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features),
                        dtype=dtype, device=device)
        )
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma,
                                    a=-3*sigma, b=3*sigma)

    def forward(self, x: Float[Tensor, " ... d_in"]
               ) -> Float[Tensor, " ... d_out"]:
        return torch.einsum("oi,...i->...o", self.weight, x)


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
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim),
                        device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, "..."]
               ) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]


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
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_sq = reduce(torch.square(x), "... d_model -> ...", "mean")
        rms = torch.sqrt(mean_sq + self.eps)
        result = x / rms[..., None] * self.weight
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
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        pre_gate = self.w1(x)
        gate = pre_gate * torch.sigmoid(pre_gate)
        up_proj = self.w3(x)
        out = self.w2(up_proj * gate)
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


def softmax(
    xs: Float[Tensor, "..."],
    dim: int = -1
) -> Float[Tensor, "..."]:
    maxs = torch.max(xs, dim, keepdim=True).values
    # maxs = torch.where(torch.isneginf(maxs), 1e-9, maxs)
    exp_xs = torch.exp(xs - maxs)
    denom = exp_xs.sum(dim, keepdim=True)
    return exp_xs / denom


def log_softmax(
    xs: Float[Tensor, "..."],
    dim: int = -1
) -> Float[Tensor, "..."]:
    max_logits = torch.max(xs, dim, keepdim=True).values
    log_denom = torch.log(torch.exp(xs - max_logits).sum(dim, keepdim=True))
    return xs - max_logits - log_denom


def scaled_dot_product_attn(
    Q: Float[Tensor, "... seq_q d_k"],
    K: Float[Tensor, "... seq_k d_k"],
    V: Float[Tensor, "... seq_k d_v"],
    mask: Bool[Tensor, "... seq_q seq_k"] | None = None
) -> Float[Tensor, "... seq_q d_v"]:
    d_k = Q.shape[-1]
    assert K.shape[-1] == d_k
    attn_scores = einsum(Q, K / math.sqrt(d_k),
        "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    if mask is not None:
        attn_scores.masked_fill_(~mask, -torch.inf)
    attn_weights = softmax(attn_scores, dim=-1)
    out = einsum(attn_weights, V,
        "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
    return out


class MultiHeadSelfAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        num_heads: int,
        *args,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert d_k % num_heads == 0, "Expecting num_heads to divide d_k"
        assert d_v % num_heads == 0, "Expecting num_heads to divide d_v"
        self.q_proj = Linear(d_model, d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_v, device=device, dtype=dtype)
        self.output_proj = Linear(d_v, d_model, device=device, dtype=dtype)
        self.rope = rope
        self.num_heads = num_heads
        self.device = device

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        mask: Int[Tensor, "... seq_q seq_k"] | None = None,
        pos_ids: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq d_model"]:
        seq_len = x.shape[-2]
        Q = rearrange(self.q_proj(x),
            "... seq (h d_kh) -> ... h seq d_kh", h=self.num_heads)
        K = rearrange(self.k_proj(x),
            "... seq (h d_kh) -> ... h seq d_kh", h=self.num_heads)
        if self.rope is not None:
            if pos_ids is None:
                pos_ids = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, pos_ids)
            K = self.rope(K, pos_ids)
        V = rearrange(self.v_proj(x),
            "... seq (h d_vh) -> ... h seq d_vh", h=self.num_heads)
        causal_mask = torch.tril(torch.ones(
            (seq_len, seq_len), device=self.device, dtype=torch.bool))
        # Do not attend for q->k if q < k
        # 1 0 0
        # 1 1 0
        # 1 1 1
        if mask is not None:
            mask = mask[..., None, :, :]
            mask = mask & causal_mask
        else:
            mask = causal_mask
        attn_values = scaled_dot_product_attn(Q, K, V, mask)
        out = self.output_proj(rearrange(attn_values,
            "... h seq d_vh -> ... seq (h d_vh)"))
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *args,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert d_model % num_heads == 0, "Expecting num_heads to divide d_model"
        self.attn = MultiHeadSelfAttn(
            d_model, d_k=d_model, d_v=d_model, num_heads=num_heads,
            rope=RoPE(d_model // num_heads, theta, max_seq_len,
                      device=device, dtype=dtype),
            device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        mask: Int[Tensor, "... seq_q seq_k"] | None = None,
        pos_ids: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq d_model"]:
        attn_resid = self.attn(self.ln1(x), mask, pos_ids)
        x = x + attn_resid
        ff_resid = self.ffn(self.ln2(x))
        x = x + ff_resid
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        *args,
        num_layers: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, num_heads=num_heads, theta=theta,
                max_seq_len=max_seq_len, d_ff=d_ff,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: Int[Tensor, "... seq"],
        mask: Int[Tensor, "... seq_q seq_k"] | None = None,
        pos_ids: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq d_model"]:
        h = self.token_embeddings(x)
        for transformer_block in self.layers:
            h = transformer_block(h, mask, pos_ids)
        logits = self.lm_head(self.ln_final(h))
        return logits
