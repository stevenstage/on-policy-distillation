"""
Training algorithms
====================
  sft_train          – Supervised Fine-Tuning (behavioral cloning on expert data)
  online_rl_train    – Online REINFORCE
  offline_opd_train  – Offline On-Policy Distillation  (pre-computed teacher labels)
  online_opd_train   – Online OPD  (fresh rollouts + live oracle) [upper bound]
  dagger_opd_train   – DAgger-style patch: periodic refresh of offline dataset
"""
 
from __future__ import annotations
 
import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
 
from env    import KGQAEnv
from model  import PolicyNet
from collect import (Trajectory, OfflineDataset,
                     rollout, collect_oracle_trajectories)
 
 
# ── SFT ───────────────────────────────────────────────────────────────────────
 
def sft_train(
    env:        KGQAEnv,
    n_expert:   int   = 200,
    epochs:     int   = 40,
    lr:         float = 1e-3,
    batch_size: int   = 64,
    verbose:    bool  = False,
) -> Tuple[PolicyNet, List[float]]:
    """
    Behavioral cloning on oracle trajectories.
    Loss: cross-entropy( student, oracle_action ) – no advantage weighting.
    """
    net   = PolicyNet(env.obs_dim, env.n_actions)
    opt   = torch.optim.Adam(net.parameters(), lr=lr)
    trajs = collect_oracle_trajectories(env, n=n_expert, eps=0.0)
 
    # Flatten to (obs, action) pairs
    obs_list, act_list = [], []
    for t in trajs:
        obs_list.extend(t.obs)
        act_list.extend(t.actions)
    X = torch.FloatTensor(np.stack(obs_list))
    Y = torch.LongTensor(act_list)
 
    losses = []
    for ep in range(epochs):
        perm  = torch.randperm(len(X))
        ep_loss = 0.0
        for start in range(0, len(X), batch_size):
            idx = perm[start:start + batch_size]
            log_p = net(X[idx])
            loss  = F.nll_loss(log_p, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        losses.append(ep_loss / len(X))
        if verbose and (ep + 1) % 10 == 0:
            print(f"  SFT epoch {ep+1}/{epochs}  loss={losses[-1]:.4f}")
    return net, losses
 
 
# ── Online RL (REINFORCE) ─────────────────────────────────────────────────────
 
def online_rl_train(
    env:         KGQAEnv,
    n_episodes:  int   = 3000,
    lr:          float = 5e-4,
    batch_size:  int   = 32,     # episodes per update
    gamma:       float = 1.0,
    verbose:     bool  = False,
) -> Tuple[PolicyNet, List[float]]:
    """
    REINFORCE with whitened per-step returns.
    Loss: -Â_t * log π(a_t | s_t)
    """
    net     = PolicyNet(env.obs_dim, env.n_actions)
    opt     = torch.optim.Adam(net.parameters(), lr=lr)
    rewards = []
    
    policy_fn = lambda obs: net.act(obs)
 
    batch_trajs: List[Trajectory] = []
    ep = 0
    while ep < n_episodes:
        qid  = int(np.random.randint(env.N_P))
        traj = rollout(env, policy_fn, person_id=qid)
        batch_trajs.append(traj)
        ep  += 1
 
        if len(batch_trajs) == batch_size or ep == n_episodes:
            _rl_update(net, opt, batch_trajs, gamma=gamma)
            rewards.extend([t.total_reward for t in batch_trajs])
            if verbose and ep % 300 == 0:
                print(f"  RL ep {ep}/{n_episodes}  "
                      f"success={np.mean([t.success for t in batch_trajs[-batch_size:]]):.2f}")
            batch_trajs = []
    return net, rewards
 
 
def _rl_update(net, opt, trajs: List[Trajectory], gamma=1.0):
    all_obs, all_acts, all_adv = [], [], []
    # Collect returns across all trajectories
    all_returns = []
    for t in trajs:
        g = t.returns(gamma)
        all_returns.extend(g.tolist())
    mean_r = float(np.mean(all_returns))
    std_r  = float(np.std(all_returns))
    # When every trajectory in the batch has the same return (e.g. all
    # succeed), std → 0 and naive whitening explodes the advantage, which can
    # destroy a good policy in a single step.  Guard with a floor and zero out
    # the (uninformative) advantages in that degenerate case.
    if std_r < 1e-4:
        mean_r, std_r = 0.0, 1.0   # advantages ≈ 0 → near-no-op update
    else:
        std_r += 1e-8
 
    for t in trajs:
        g   = t.returns(gamma)
        adv = (g - mean_r) / std_r
        all_obs.extend(t.obs)
        all_acts.extend(t.actions)
        all_adv.extend(adv.tolist())
 
    X   = torch.FloatTensor(np.stack(all_obs))
    A   = torch.LongTensor(all_acts)
    ADV = torch.FloatTensor(all_adv)
 
    log_p = net(X)                              # [N, n_actions]
    sel   = log_p[torch.arange(len(A)), A]      # [N] – chosen action log-prob
    loss  = -(ADV * sel).mean()
 
    opt.zero_grad(); loss.backward(); opt.step()
 
 
# ── Offline OPD ───────────────────────────────────────────────────────────────
 
def offline_opd_train(
    env:         KGQAEnv,
    dataset:     OfflineDataset,
    sft_net:     PolicyNet,  # Initialize from SFT model!
    epochs:      int   = 40,
    lr:          float = 1e-3,
    batch_size:  int   = 64,
    adv_clip:    float = 2.0,
    verbose:     bool  = False,
) -> Tuple[PolicyNet, List[float], List[float]]:
    """
    Offline On-Policy Distillation (Lightning OPD).

    Key: Student is initialized from π_ref (SFT model), not from scratch!
    """
    # Initialize from SFT model (π_ref)
    net = sft_net.copy()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    X_all   = torch.FloatTensor(np.stack(dataset.obs))
    A_all  = torch.LongTensor(dataset.actions)  # actions from π_ref rollouts
    LogPT_all = torch.FloatTensor(dataset.teacher_logprobs)  # precomputed log π_T(a_t|s_t)

    losses, off_support_hist = [], []

    for ep in range(epochs):
        perm     = torch.randperm(len(X_all))
        ep_loss  = 0.0
        for start in range(0, len(X_all), batch_size):
            idx    = perm[start:start + batch_size]
            log_p_student  = net(X_all[idx])  # [B, n_actions]
            actions = A_all[idx]  # actions from π_ref
            log_p_T = LogPT_all[idx]  # precomputed log π_T(a_t|s_t)

            # Student log-prob on the same action
            log_p_theta = log_p_student[torch.arange(len(actions)), actions]

            # OPD advantage: A_t = log π_T(a_t|s_t) - log π_θ(a_t|s_t)
            advantage = log_p_T - log_p_theta
            advantage = advantage.clamp(-adv_clip, adv_clip)

            # OPD objective: maximize E[ A_t * log π_θ(a_t|s_t) ]
            # Equivalent loss to minimize: -E[ A_t * log π_θ(a_t|s_t) ]
            loss = -(advantage.detach() * log_p_theta).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        losses.append(ep_loss / len(X_all))

        # Measure off-support ratio every 5 epochs
        if (ep + 1) % 5 == 0:
            osr = _off_support_ratio(env, net, dataset.support_keys, n=50)
            off_support_hist.append(osr)
            if verbose:
                print(f"  Offline OPD epoch {ep+1}/{epochs}  "
                      f"loss={losses[-1]:.4f}  off-support={osr:.3f}")

    return net, losses, off_support_hist
 
 
# ── Online OPD ────────────────────────────────────────────────────────────────
 
def online_opd_train(
    env:         KGQAEnv,
    sft_net:     PolicyNet,  # Initialize from SFT model
    n_episodes:  int   = 3000,
    lr:          float = 5e-4,
    batch_size:  int   = 32,
    gamma:       float = 1.0,
    adv_clip:    float = 2.0,
    verbose:     bool  = False,
) -> Tuple[PolicyNet, List[float]]:
    """
    Online OPD (upper bound).
    Roll out current student, compute advantages, then train on ORACLE actions
    (not student's own actions – that's the key difference from online RL).
    """
    net     = sft_net.copy()  # Initialize from SFT
    opt     = torch.optim.Adam(net.parameters(), lr=lr)
    rewards = []
    policy_fn = lambda obs: net.act(obs)
 
    batch_trajs: List[Trajectory] = []
    ep = 0
    while ep < n_episodes:
        qid  = int(np.random.randint(env.N_P))
        traj = rollout(env, policy_fn, person_id=qid)
        batch_trajs.append(traj)
        ep  += 1
 
        if len(batch_trajs) == batch_size or ep == n_episodes:
            _opd_online_update(net, opt, batch_trajs, env, gamma, adv_clip)
            rewards.extend([t.total_reward for t in batch_trajs])
            if verbose and ep % 300 == 0:
                print(f"  Online OPD ep {ep}/{n_episodes}  "
                      f"success={np.mean([t.success for t in batch_trajs[-batch_size:]]):.2f}")
            batch_trajs = []
    return net, rewards
 
 
def _opd_online_update(net, opt, trajs, env, gamma, adv_clip):
    """
    Online OPD update (paper-faithful).

    For each step in the student's OWN rollout, the advantage is the
    per-token OPD advantage from the paper:

        A_t = log π_T(a_t | s_t) − log π_θ(a_t | s_t)        (stop-grad, clipped)

    where a_t is the student's sampled action (NOT the oracle action) and
    π_T is the deterministic oracle teacher:
        log π_T(a_t) = 0      if a_t == oracle(s_t)
        log π_T(a_t) = −10    otherwise
    The loss is −E[ A_t · log π_θ(a_t|s_t) ], i.e. minimising reverse-KL.
    This is the ONLY thing that differs from online RL: the advantage is the
    teacher/student log-prob gap, never the task reward.
    """
    all_obs, all_acts, all_teacher_lp = [], [], []
    for traj in trajs:
        # Restore this episode's graph so oracle(s_t) is faithful.
        env.reset(traj.state_keys[0][0], graph=traj.graph)
        for t_idx, act in enumerate(traj.actions):
            oa = env.oracle_action()
            all_obs.append(traj.obs[t_idx])
            all_acts.append(act)
            all_teacher_lp.append(0.0 if act == oa else -10.0)
            env.step(act)   # advance with the student's action

    X   = torch.FloatTensor(np.stack(all_obs))
    A   = torch.LongTensor(all_acts)
    LPT = torch.FloatTensor(all_teacher_lp)

    log_p       = net(X)
    log_p_theta = log_p[torch.arange(len(A)), A]          # log π_θ(a_t|s_t)
    advantage   = (LPT - log_p_theta).clamp(-adv_clip, adv_clip).detach()
    loss        = -(advantage * log_p_theta).mean()
    opt.zero_grad(); loss.backward(); opt.step()
 
 
# ── DAgger-style Offline OPD (patch) ─────────────────────────────────────────
 
def dagger_opd_train(
    env:          KGQAEnv,
    dataset:      OfflineDataset,
    sft_net:      PolicyNet,  # Initialize from SFT
    epochs:       int   = 40,
    lr:           float = 1e-3,
    batch_size:   int   = 64,
    adv_clip:     float = 2.0,
    refresh_every: int  = 8,    # epochs between DAgger refreshes
    refresh_eps:   int  = 80,   # student rollouts per refresh
    verbose:       bool = False,
) -> Tuple[PolicyNet, List[float], List[float]]:
    """
    DAgger-style patch for Offline OPD.

    Every `refresh_every` epochs, roll out the current student to collect
    fresh trajectories, annotate with oracle teacher, and add to training
    data.  This reduces the off-support ratio and addresses distribution
    shift – the core weakness of naive offline OPD in multi-turn agentic
    settings.
    """
    net = sft_net.copy()  # Initialize from SFT
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Aggregated dataset of (state, teacher_action).  DAgger does behaviour
    # cloning on the TEACHER action over the state distribution the student
    # actually visits — so we only need obs + oracle action, no advantages.
    X_dyn   = list(dataset.obs)
    TA_dyn  = list(dataset.teacher_actions)
    support = set(dataset.support_keys)

    losses, off_support_hist = [], []

    for ep in range(epochs):
        # ── DAgger refresh ──────────────────────────────────────────────────
        if ep > 0 and ep % refresh_every == 0:
            policy_fn = lambda obs: net.act(obs)
            for _ in range(refresh_eps):
                qid  = int(np.random.randint(env.N_P))
                traj = rollout(env, policy_fn, person_id=qid)
                # Annotate the student's visited states with the oracle action.
                # Restore this episode's graph so oracle(s_t) is faithful.
                env.reset(traj.state_keys[0][0], graph=traj.graph)
                for t_idx, act in enumerate(traj.actions):
                    oa = env.oracle_action()
                    X_dyn.append(traj.obs[t_idx])
                    TA_dyn.append(oa)
                    support.add(traj.state_keys[t_idx])
                    env.step(act)

        # ── Training step (behaviour cloning on teacher actions) ─────────────
        X_t   = torch.FloatTensor(np.stack(X_dyn))
        TA_t  = torch.LongTensor(TA_dyn)

        perm    = torch.randperm(len(X_t))
        ep_loss = 0.0
        for start in range(0, len(X_t), batch_size):
            idx   = perm[start:start + batch_size]
            log_p = net(X_t[idx])
            loss  = F.nll_loss(log_p, TA_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
        losses.append(ep_loss / len(X_t))
 
        # Off-support ratio on fresh student rollouts
        if (ep + 1) % 5 == 0:
            osr = _off_support_ratio(env, net, support, n=50)
            off_support_hist.append(osr)
            if verbose:
                print(f"  DAgger OPD epoch {ep+1}/{epochs}  "
                      f"loss={losses[-1]:.4f}  off-support={osr:.3f}  "
                      f"dataset_size={len(X_dyn)}")
 
    return net, losses, off_support_hist
 
 
# ── Utility ───────────────────────────────────────────────────────────────────
 
def _off_support_ratio(env: KGQAEnv, net: PolicyNet,
                        support_keys: set, n: int = 100) -> float:
    """
    Roll out policy for n episodes; count fraction of visited state_keys
    NOT present in support_keys (the offline dataset's state coverage).
    """
    total, out = 0, 0
    policy_fn  = lambda obs: net.act(obs, greedy=True)
    for _ in range(n):
        qid  = int(np.random.randint(env.N_P))
        traj = rollout(env, policy_fn, person_id=qid)
        for sk in traj.state_keys:
            total += 1
            if sk not in support_keys:
                out += 1
    return out / max(total, 1)