
def print_flop_accounting_report(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
) -> None:
    trainable_params = (
        vocab_size * d_model  # Embedding
        + num_layers * (
            d_model  # Layer norm 1
            + 4 * d_model * d_model  # Q,K,V and output projections
            # assuming all heads together recover the model dimension
            + d_model  # Layer norm 2
            + 2 * d_model * d_ff + d_ff * d_model
            # assuming gated linear unit
        )
        + d_model  # Layer norm
        + d_model * vocab_size  # LM head
    )
    print(f"trainable_params={trainable_params/1e9:,.2f}B")

    # Assuming 32bits = 4 bytes per param as instructed
    params_bytes = trainable_params * 4
    print(f"params_bytes={params_bytes / 1_024**3:,.2f}GiB")


    # FLOPs accounting
    def gemm_flops(d_in, d_shared, d_out): return 2 * d_in * d_shared * d_out
    # Assume batch_size = 1
    # Embedding is a gather operation not GEMM, ignoring
    attn_proj_flops = (
        # Q,K,V and output projections
        4 * gemm_flops(context_length, d_model, d_model)
    )
    attn_weights_flops = (
        # Attention scores
        gemm_flops(num_heads * context_length, d_model // num_heads, context_length)
        # Attention weighted values
        + gemm_flops(num_heads * context_length, context_length, d_model // num_heads)
    )
    attn_flops = attn_proj_flops + attn_weights_flops
    ffn_flops = (
        2 * gemm_flops(context_length, d_model, d_ff)
        + gemm_flops(context_length, d_ff, d_model)
    )
    lm_flops = gemm_flops(context_length, d_model, vocab_size)
    forward_matmul_flops = num_layers * (attn_flops + ffn_flops) + lm_flops
    print("================")
    # print(f"{num_layers*attn_flops/forward_matmul_flops=:.2%}")
    print(f"{num_layers*attn_proj_flops/forward_matmul_flops=:.2%}")
    print(f"{num_layers*attn_weights_flops/forward_matmul_flops=:.2%}")
    print(f"{num_layers*ffn_flops/forward_matmul_flops=:.2%}")
    print(f"{lm_flops/forward_matmul_flops=:.2%}")
    print(f"forward_matmul_flops={forward_matmul_flops/1e12:,.2f}TFLOPs")


print("\n\nGPT-2 small")
# GPT-2 small (12 layers, 768 d_model, 12 heads)
print_flop_accounting_report(
    vocab_size = 50_257,
    context_length = 1_024,
    num_layers = 12,
    d_model = 768,
    num_heads = 12,
    d_ff = 64 * ((768 * 8 / 3) // 64)
)

print("\n\nGPT-2 medium")
# GPT-2 medium (24 layers, 1024 d_model, 16 heads)
print_flop_accounting_report(
    vocab_size = 50_257,
    context_length = 1_024,
    num_layers = 24,
    d_model = 1_024,
    num_heads = 16,
    d_ff = 64 * ((1_024 * 8 / 3) // 64)
)

print("\n\nGPT-2 large")
# GPT-2 large (36 layers, 1280 d_model, 20 heads)
print_flop_accounting_report(
    vocab_size = 50_257,
    context_length = 1_024,
    num_layers = 36,
    d_model = 1_280,
    num_heads = 20,
    d_ff = 64 * ((1_280 * 8 / 3) // 64)
)

print("\n\nGPT-2 XL")
print_flop_accounting_report(
    vocab_size = 50_257,
    context_length = 1_024,
    num_layers = 48,
    d_model = 1_600,
    num_heads = 25,
    d_ff = 4_288
)

print("\n\nGPT-2 XL 16k context")
print_flop_accounting_report(
    vocab_size = 50_257,
    context_length = 16*1_024,
    num_layers = 48,
    d_model = 1_600,
    num_heads = 25,
    d_ff = 4_288
)


"""
ffn + attn projection: O(context_length * d_model**2)
attn weights: O(context_length**2 * d_model)
As model size increases (width) ffn matmuls and attn projections
become a higher and higher % of FLOPs.
With increasing context length attn computation grows in proportion
by similar quadratic scaling.
"""
