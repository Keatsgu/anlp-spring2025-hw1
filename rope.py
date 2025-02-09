from typing import Tuple
import torch

# Helper function to reshape the frequency tensor for broadcasting  with the input tensor
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim

    # Verify that x has at least 2 dimensions.
    assert 0 <= 1 < ndim, "x must have at least 2 dimensions."

    # Ensure that freqs_cis has the shape (x.shape[1], x.shape[-1]).
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        f"freqs_cis must have shape ({x.shape[1]}, {x.shape[-1]}), "
        f"but got {freqs_cis.shape}."
    )

    # Keep the second and last dimensions, set others to 1
    shape = []
    for i, d in enumerate(x.shape):
        if i == 1 or i == ndim - 1:
            shape.append(d)
        else:
            shape.append(1)

    return freqs_cis.view(shape)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    # Compute theta_i for each dimension
    d = head_dim
    i = torch.arange(0, d // 2, dtype=torch.float32, device=device)
    theta_i = theta ** (-2 * i / d)

    # Generate positions up to max_seq_len and compute frequencies
    m = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = m.unsqueeze(-1) * theta_i.unsqueeze(0)  # (max_seq_len, d//2)
    freqs = freqs[:seqlen]  # (seqlen, d//2)

    # Compute cos and sin values
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # Reshape for broadcasting with query and key
    cos_q = reshape_for_broadcast(cos, query_real)
    sin_q = reshape_for_broadcast(sin, query_real)
    cos_k = reshape_for_broadcast(cos, key_real)
    sin_k = reshape_for_broadcast(sin, key_real)

    # Apply rotation to real and imaginary parts
    query_rotated_real = query_real * cos_q - query_imag * sin_q
    query_rotated_imag = query_real * sin_q + query_imag * cos_q
    key_rotated_real = key_real * cos_k - key_imag * sin_k
    key_rotated_imag = key_real * sin_k + key_imag * cos_k

    # Combine back to original shape and cast to original dtype
    original_dtype = query.dtype
    query_out = torch.stack([query_rotated_real, query_rotated_imag], dim=-1).flatten(-2, -1).to(original_dtype)
    key_out = torch.stack([key_rotated_real, key_rotated_imag], dim=-1).flatten(-2, -1).to(original_dtype)
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out