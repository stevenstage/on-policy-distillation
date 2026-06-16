"""Evaluation utilities."""

import numpy as np
from typing import Dict
from collect import rollout
from env import KGQAEnv
from model import PolicyNet


def evaluate_policy(env: KGQAEnv, net: PolicyNet, n: int = 500,
                    greedy: bool = True) -> Dict:
    """
    Evaluate a trained policy on n episodes.

    Returns:
        Dictionary with success_rate, avg_steps, avg_reward, etc.
    """
    policy_fn = lambda obs: net.act(obs, greedy=greedy)

    successes = []
    steps = []
    rewards = []

    for _ in range(n):
        qid = int(np.random.randint(env.N_P))
        traj = rollout(env, policy_fn, person_id=qid)

        successes.append(traj.success)
        steps.append(traj.length)
        rewards.append(traj.total_reward)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }
