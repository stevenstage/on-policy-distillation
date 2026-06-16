"""Policy network: small MLP  obs -> action log-probabilities."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int,
                 hidden: tuple = (128, 64)):
        super().__init__()
        dims   = [obs_dim, *hidden, n_actions]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-softmax over actions."""
        return F.log_softmax(self.net(x), dim=-1)

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).exp()

    # ── inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False,
            eps_random: float = 0.0) -> int:
        """Sample an action from the policy (optionally ε-greedy)."""
        if eps_random > 0 and np.random.rand() < eps_random:
            return int(np.random.randint(
                self.net[-1].out_features))   # type:ignore
        x        = torch.FloatTensor(obs).unsqueeze(0)
        log_prob = self.forward(x).squeeze(0)
        if greedy:
            return int(log_prob.argmax())
        return int(torch.distributions.Categorical(logits=log_prob).sample())

    def copy(self) -> "PolicyNet":
        import copy
        return copy.deepcopy(self)