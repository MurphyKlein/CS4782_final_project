import torch


def make_patches(x: torch.Tensor, P: int, S: int) -> torch.Tensor:
    """
    Segment a batch of channel-first time series into overlapping patches.

    Args:
        x: Tensor with shape (B, M, L), batch by channels by look-back.
        P: Patch length.
        S: Stride. When P > S, patches overlap.

    Returns:
        Tensor with shape (B*M, N, P), where
        N = floor((L - P) / S) + 2.
    """
    B, M, _ = x.shape

    pad = x[:, :, -1:].expand(-1, -1, S)
    x_pad = torch.cat([x, pad], dim=-1)

    patches = x_pad.unfold(dimension=2, size=P, step=S)
    N = patches.shape[2]

    return patches.reshape(B * M, N, P)

