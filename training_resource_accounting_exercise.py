
def print_train_acct_report(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int
):
    d_ff = (8 * d_model) // 3
    param_count = (
        vocab_size * d_model  # Embedding
        + num_layers * (  # Transformer blocks
            2 * d_model  # 2 RMSNorm's
            + 4 * d_model * d_model  # QKVO proj
            + 3 * d_model * d_ff  # SwiGLU weights
        )
        + d_model  # RMSNorm
        + d_model * vocab_size  # LM head
    )
    input_count = batch_size * context_length  # Minor but needs to be stored on device for embedding's backward
    activ_count = (
        batch_size * context_length * d_model  # Initial embeddings
        + batch_size * num_layers * (
            2 * context_length * d_model  # 2 RMSNorm
            + 3 * context_length * d_model  # QKV proj outputs
            + 2 * num_heads * context_length * context_length  # QK' scores + weights
            + context_length * d_model  # weighted sum of values
            # O proj (sub-layer) output
            + context_length * d_model
            + 4 * context_length * d_ff  # Gate and up-projected SwiGLU activations + e/w product + e/w nonlinearity
            + context_length * d_model  # FFN output
        )
        + batch_size * context_length * d_model  # Final RMSNorm
        + 2 * batch_size * context_length * vocab_size  # Logits + logprobs
        + 1  # Scalar loss
    )
    grad_count = param_count  # Each param gets a grad; intermediate activation grads are not needed simultaneously and compute one by one (or a few by a few in backward)
    # Say conservatively that we allocate *all* param gradients when all intermediate activations are still alive; in practice we'd save some by having deallocated e.g. deeper layer activations by the time we reach earlier layers' param gradients, assuming no gradient accummulation enabled
    optim_state_count = 2 * param_count  # Each diff'ble param gets first and second moment running estimate
    total = (
        param_count
        + input_count
        + activ_count
        + grad_count
        + optim_state_count
    ) * 4  # 32 bit = 4 bytes / value

    print(f"{param_count=}")
    print(f"{activ_count=}")
    print(f"{grad_count=}")
    print(f"{optim_state_count=}")
    print(f"{total=:,} memory")

    # Forward flops
    forward_flops = 2 * (
        # embedding is lookup, no flops
        batch_size * num_layers * (
            4 * context_length * d_model * d_model  # qkvo
            + num_heads * context_length * (d_model // num_heads) * context_length
            + num_heads * context_length * context_length * (d_model // num_heads)
            + 3 * context_length * d_model * d_ff
        )
        + batch_size * context_length * d_model * vocab_size
    )
    print(f"{forward_flops=:,}")


print_train_acct_report(
    batch_size=1,
    vocab_size = 50_257,
    context_length = 1_024,
    num_layers = 48,
    d_model = 1_600,
    num_heads = 25,
)

tflops_per_pass = 3 * 2 * 1_024 * 1.753 # Fwd + bwd, *+, batch
total_required_tflops = 400_000 * tflops_per_pass
assumed_tflop_per_s = 495 * 0.5
total_s = total_required_tflops / assumed_tflop_per_s
print(f"f{total_s/60/60=:,.1f}")  # 2.4 hours?