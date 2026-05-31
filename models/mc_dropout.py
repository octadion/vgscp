"""MC-dropout inference — Section 6, 13.

K stochastic passes with dropout ENABLED at eval time. The K passes are BATCHED (the input is
tiled K times in the batch dimension) rather than looped in python (Section 13). Returns the
per-pass probabilities (K, N, C) which the mcdropout signal consumes.
"""
from __future__ import annotations

import numpy as np


def enable_dropout(net):
    """Set only Dropout layers to train mode; everything else stays in eval mode."""
    import torch.nn as nn

    for m in net.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    return net


def mc_dropout_probs_batched(classifier, x, k_passes: int, perf_ctx):
    """Return (K, B, C) probabilities for a batch x by tiling K passes into the batch dim.

    Tiling exploits that dropout masks are sampled independently per row, so K replicas of each
    sample in one forward pass give K independent stochastic predictions — no python loop.
    """
    import torch

    enable_dropout(classifier.net)
    b = x.shape[0]
    # tile: (K*B, ...) — each sample repeated K times, contiguous blocks per pass
    x_tiled = x.repeat(k_passes, *([1] * (x.dim() - 1)))
    if perf_ctx.channels_last:
        x_tiled = x_tiled.to(memory_format=torch.channels_last)
    with torch.autocast(device_type="cuda", dtype=perf_ctx.amp_dtype, enabled=perf_ctx.use_amp):
        logits, _ = classifier.logits_and_features(x_tiled)
    probs = torch.softmax(logits.float(), dim=1)
    return probs.view(k_passes, b, -1)
