"""Trajectory collection and helper utilities."""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional

from env import KGQAEnv


# ── Trajectory dataclass ──────────────────────────────────────────────────────

class Trajectory:
    """Single rollout episode."""

    def __init__(self):
        self.obs:        List[np.ndarray] = []
        self.actions:    List[int]        = []
        self.rewards:    List[float]      = []
        self.infos:      List[dict]       = []
        self.state_keys: List[tuple]      = []  # for off-support detection
        self.graph:      Optional[tuple]  = None  # this episode's KG snapshot

    def append(self, obs, action, reward, info, state_key):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info)
        self.state_keys.append(state_key)

    # ── derived quantities ────────────────────────────────────────────────────

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def success(self) -> bool:
        return self.total_reward > 0

    @property
    def length(self) -> int:
        return len(self.actions)

    def returns(self, gamma: float = 1.0) -> np.ndarray:
        """G_t = Σ_{t'≥t} γ^{t'-t} r_{t'}"""
        R   = np.array(self.rewards, dtype=np.float32)
        G   = np.zeros_like(R)
        acc = 0.0
        for t in reversed(range(len(R))):
            acc  = R[t] + gamma * acc
            G[t] = acc
        return G

    def advantages(self, baseline: float = 0.5,
                   gamma: float = 1.0) -> np.ndarray:
        return self.returns(gamma) - baseline

    def teacher_actions(self, env_snapshot: KGQAEnv) -> List[int]:
        """
        Recompute oracle action at every step by replaying the episode
        in a copy of the environment.
        (Only used for pre-computing teacher labels in offline OPD.)
        """
        # We replay from stored obs / infos instead of a live env copy
        # because we stored state_keys which encode full state
        raise NotImplementedError("Use precompute_teacher_labels instead")


# ── Rollout helpers ───────────────────────────────────────────────────────────

def rollout(env: KGQAEnv, policy_fn, person_id: Optional[int] = None,
            eps_random: float = 0.0) -> Trajectory:
    """
    Collect one episode.

    policy_fn: callable(obs) -> action_int
    eps_random: if > 0, use ε-random override
    """
    traj = Trajectory()
    obs  = env.reset(person_id)
    traj.graph = env.get_graph()   # snapshot for faithful replay
    done = False
    while not done:
        sk     = env.state_key()
        action = policy_fn(obs)
        if eps_random > 0 and np.random.rand() < eps_random:
            action = int(np.random.randint(env.n_actions))
        obs_next, reward, done, info = env.step(action)
        traj.append(obs, action, reward, info, sk)
        obs = obs_next
    return traj


def collect_oracle_trajectories(env: KGQAEnv, n: int = 200,
                                 eps: float = 0.0) -> List[Trajectory]:
    """
    Collect n trajectories using oracle (± ε random noise).
    eps=0   → perfect expert (all succeed, 3 steps)
    eps>0   → noisy expert   (sometimes suboptimal)
    """
    trajs = []
    for _ in range(n):
        qid  = int(np.random.randint(env.N_P))
        traj = rollout(env, lambda obs: env.oracle_action(),
                       person_id=qid, eps_random=eps)
        trajs.append(traj)
    return trajs


# ── Offline dataset ───────────────────────────────────────────────────────────

class OfflineDataset:
    """
    Pre-collected trajectory dataset with teacher labels.

    For Lightning OPD:
    - teacher_action[i] = oracle action at step i (deterministic oracle)
    - teacher_logprob[i] = log π_T(teacher_action[i] | state[i])
    - support_keys = set of all state_keys seen in this dataset

    Note: OPD advantage A_t = log π_T - log π_θ is computed dynamically
    during training, not stored here. We only store log π_T.
    """

    def __init__(self, trajectories: List[Trajectory], env: KGQAEnv):
        self.trajs  = trajectories
        self.env    = env
        self._build()

    def _build(self):
        """Flatten trajectories; recompute oracle labels via faithful replay."""
        self.obs:              List[np.ndarray] = []
        self.actions:          List[int]        = []   # actions from π_ref rollouts
        self.teacher_actions:  List[int]        = []   # oracle action at each s_t
        self.teacher_logprobs: List[float]      = []   # log π_T(a_t | s_t)
        self.advantages:       List[float]      = []   # reward-based, baselined
        self.support_keys:     set              = set()

        # Global baseline = mean trajectory reward (for DAgger reward-advantages)
        baseline = float(np.mean([t.total_reward for t in self.trajs])) \
            if self.trajs else 0.0

        for traj in self.trajs:
            env = self.env
            # Restore THIS episode's graph so the replay is faithful.
            env.reset(traj.state_keys[0][0], graph=traj.graph)
            adv = traj.advantages(baseline=baseline)

            for t_idx in range(traj.length):
                action  = traj.actions[t_idx]
                oracle  = env.oracle_action()    # oracle at current replayed state

                self.obs.append(traj.obs[t_idx])
                self.actions.append(action)
                self.teacher_actions.append(oracle)
                # Deterministic oracle teacher: log π_T = 0 on the oracle action,
                # large-negative otherwise (prob ≈ 0).
                self.teacher_logprobs.append(0.0 if action == oracle else -10.0)
                self.advantages.append(float(adv[t_idx]))
                self.support_keys.add(traj.state_keys[t_idx])

                env.step(action)                 # advance with the π_ref action

    def __len__(self) -> int:
        return len(self.obs)

    def sample_batch(self, size: int):
        """Random mini-batch."""
        idx = np.random.randint(0, len(self.obs), size)
        return (
            np.stack([self.obs[i]             for i in idx]),
            np.array([self.teacher_actions[i] for i in idx]),
            np.array([self.advantages[i]      for i in idx], dtype=np.float32),
            [self.obs[i]                      for i in idx],   # for key lookup
            idx,
        )