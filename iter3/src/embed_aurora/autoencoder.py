"""MLP autoencoder: 28 -> 16 -> 8 -> 16 -> 28."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


LATENT_DIM = 8


class TrajAE(nn.Module):
    def __init__(self, in_dim: int = 28, hidden: int = 16, latent: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decoder(z)
        return recon, z

    @torch.no_grad()
    def embed(self, x: np.ndarray) -> np.ndarray:
        """x: (n, in_dim) or (in_dim,). Returns (n, latent) or (latent,)."""
        single = x.ndim == 1
        t = torch.as_tensor(x, dtype=torch.float32)
        if single:
            t = t.unsqueeze(0)
        z = self.encode(t).cpu().numpy()
        return z[0] if single else z
