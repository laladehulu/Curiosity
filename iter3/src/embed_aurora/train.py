"""Fit / refit the AURORA-style AE on accumulated pooled features.

Container reset: every time we refit, we re-init the AE from scratch (so
the latent space is unconstrained by prior parameters), train for `epochs`
on all features so far, then return the fitted model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.embed_aurora.autoencoder import TrajAE


def fit_autoencoder(
    features: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> TrajAE:
    """Train AE on (n, in_dim) features. Returns the fitted model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TrajAE(in_dim=features.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x = torch.as_tensor(features, dtype=torch.float32)
    n = x.shape[0]
    if batch_size is None or batch_size >= n:
        batches = [x]
    else:
        perm = torch.randperm(n)
        batches = [x[perm[i:i + batch_size]] for i in range(0, n, batch_size)]
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb in batches:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()) * xb.shape[0]
        if verbose and ((ep + 1) % 10 == 0 or ep == epochs - 1):
            print(f"  [AE] epoch {ep+1}/{epochs}  loss={ep_loss / n:.4f}")
    model.eval()
    return model
