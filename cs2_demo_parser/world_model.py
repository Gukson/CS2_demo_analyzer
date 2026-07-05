from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class WorldModelConfig:
    player_feature_dim: int
    target_feature_dim: int
    global_feature_dim: int
    future_steps: int
    max_players: int
    event_label_dim: int
    hidden_dim: int = 256
    latent_dim: int = 64
    global_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldModelConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class WorldModelCVAE(nn.Module):
    """Conditional sequential VAE for CS2 future-state prediction.

    The model observes history `[B, H, P, F]`, player masks and global features, then learns:
    - a prior `p(z | history, global)`,
    - a posterior `q(z | history, future, global)` for CVAE training,
    - future positions `[B, T, P, target_features]`,
    - future alive logits `[B, T, P]`,
    - event logits and round-winner logits as auxiliary objectives.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        history_step_dim = cfg.max_players * (cfg.player_feature_dim + 1)
        future_step_dim = cfg.max_players * (cfg.target_feature_dim + 1)

        self.global_encoder = nn.Sequential(
            nn.Linear(cfg.global_feature_dim, cfg.global_dim),
            nn.LayerNorm(cfg.global_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.history_encoder = nn.GRU(
            input_size=history_step_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.future_encoder = nn.GRU(
            input_size=future_step_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        condition_dim = cfg.hidden_dim + cfg.global_dim
        posterior_dim = condition_dim + cfg.hidden_dim
        self.prior = _GaussianHead(condition_dim, cfg.latent_dim)
        self.posterior = _GaussianHead(posterior_dim, cfg.latent_dim)

        decoder_input_dim = cfg.latent_dim + condition_dim
        self.decoder = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.position_head = nn.Linear(cfg.hidden_dim, cfg.max_players * cfg.target_feature_dim)
        self.alive_head = nn.Linear(cfg.hidden_dim, cfg.max_players)
        self.event_head = nn.Sequential(
            nn.Linear(condition_dim + cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.event_label_dim),
        )
        self.winner_head = nn.Sequential(
            nn.Linear(condition_dim + cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 3),
        )

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        global_features: torch.Tensor,
        future: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        condition = self.encode_condition(history, history_mask, global_features)
        prior_mu, prior_logvar = self.prior(condition)

        posterior_mu = posterior_logvar = None
        if self.training and future is not None and future_mask is not None:
            future_context = self.encode_future(future, future_mask)
            posterior_mu, posterior_logvar = self.posterior(torch.cat([condition, future_context], dim=-1))
            z = reparameterize(posterior_mu, posterior_logvar) if sample else posterior_mu
        else:
            z = reparameterize(prior_mu, prior_logvar) if sample else prior_mu

        decoded = self.decode(condition, z)
        aux_input = torch.cat([condition, z], dim=-1)
        return {
            "future_pos": decoded["future_pos"],
            "future_alive_logits": decoded["future_alive_logits"],
            "event_logits": self.event_head(aux_input),
            "winner_logits": self.winner_head(aux_input),
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "posterior_mu": posterior_mu if posterior_mu is not None else prior_mu,
            "posterior_logvar": posterior_logvar if posterior_logvar is not None else prior_logvar,
            "z": z,
        }

    def encode_condition(self, history: torch.Tensor, history_mask: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        batch, steps, players, features = history.shape
        mask = history_mask.unsqueeze(-1).float()
        history_input = torch.cat([history * mask, mask], dim=-1).reshape(batch, steps, players * (features + 1))
        _, hidden = self.history_encoder(history_input)
        history_context = hidden[-1]
        global_context = self.global_encoder(global_features)
        return torch.cat([history_context, global_context], dim=-1)

    def encode_future(self, future: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, players, features = future.shape
        mask = future_mask.unsqueeze(-1).float()
        future_input = torch.cat([future * mask, mask], dim=-1).reshape(batch, steps, players * (features + 1))
        _, hidden = self.future_encoder(future_input)
        return hidden[-1]

    def decode(self, condition: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = condition.shape[0]
        decoder_token = torch.cat([condition, z], dim=-1).unsqueeze(1)
        decoder_input = decoder_token.expand(batch, self.cfg.future_steps, decoder_token.shape[-1])
        decoded, _ = self.decoder(decoder_input)
        future_pos = self.position_head(decoded).reshape(
            batch, self.cfg.future_steps, self.cfg.max_players, self.cfg.target_feature_dim
        )
        future_alive_logits = self.alive_head(decoded).reshape(batch, self.cfg.future_steps, self.cfg.max_players)
        return {"future_pos": future_pos, "future_alive_logits": future_alive_logits}


class _GaussianHead(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim), nn.GELU())
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(x)
        return self.mu(hidden), self.logvar(hidden).clamp(min=-10.0, max=10.0)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def kl_divergence(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    posterior_var = torch.exp(posterior_logvar)
    prior_var = torch.exp(prior_logvar)
    kl = 0.5 * (
        prior_logvar
        - posterior_logvar
        + (posterior_var + (posterior_mu - prior_mu).pow(2)) / prior_var.clamp_min(1e-8)
        - 1.0
    )
    return kl.sum(dim=-1).mean()
