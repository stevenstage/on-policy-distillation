"""
Lightning OPD Toy Experiment
=============================
Compare three methods on multi-hop KG-QA task:
  1. SFT          – supervised fine-tuning on oracle trajectories
  2. Online RL    – REINFORCE (baseline)
  3. Offline OPD  – pre-computed teacher labels (Lightning OPD core idea)
  4. Online OPD   – fresh rollouts + live oracle (upper bound)
  5. DAgger OPD   – periodic refresh to combat distribution shift

Key research question:
  Does offline OPD degrade in multi-turn agentic tasks due to
  distribution shift when the student deviates from offline support?
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from env import KGQAEnv
from collect import collect_oracle_trajectories, OfflineDataset, rollout
from train import (sft_train, online_rl_train, offline_opd_train,
                   online_opd_train, dagger_opd_train)
from evaluate import evaluate_policy


def run_experiment(args):
    """Run full comparison experiment."""

    # Fixed per-method offsets (NOT hash(): Python salts str hashes per
    # process via PYTHONHASHSEED, which would make runs non-reproducible).
    _SEED_OFFSET = {
        "sft": 1, "dataset": 2, "online_rl": 3,
        "offline_opd": 4, "online_opd_m": 5, "dagger_opd": 6,
    }

    def reseed(tag: str):
        """Reset RNGs before each method so runs are reproducible and the
        env graph stream is identical across methods (fair comparison)."""
        s = args.seed + _SEED_OFFSET[tag]
        np.random.seed(s)
        torch.manual_seed(s)
        env._rng = np.random.RandomState(s)

    print("=" * 70)
    print("Lightning OPD Toy Experiment: Multi-hop KG-QA")
    print("=" * 70)

    # ── Environment ───────────────────────────────────────────────────────────
    env = KGQAEnv(
        n_persons=args.n_persons,
        n_companies=args.n_companies,
        n_countries=args.n_countries,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    print(f"\nEnvironment:")
    print(f"  Entities: {args.n_persons} persons, {args.n_companies} companies, "
          f"{args.n_countries} countries")
    print(f"  Action space: {env.n_actions}")
    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Max steps: {args.max_steps}")

    results = {"config": vars(args), "methods": {}}

    # ── Stage 1: SFT (required for Lightning OPD) ────────────────────────────
    print(f"\n[Stage 1] Training SFT on {args.n_offline} oracle trajectories...")
    reseed("sft")
    t0 = time.time()
    sft_net, sft_losses = sft_train(
        env, n_expert=args.n_offline, epochs=args.sft_epochs,
        lr=args.lr, batch_size=args.batch_size, verbose=True
    )
    sft_results = evaluate_policy(env, sft_net, n=args.n_eval)
    print(f"  SFT success rate: {sft_results['success_rate']:.2%}")
    print(f"  Time: {time.time()-t0:.1f}s")

    if "sft" in args.methods:
        results["methods"]["sft"] = {
            "losses": sft_losses, **sft_results, "time": time.time()-t0
        }

    # ── Stage 2: Collect OPD dataset from SFT rollouts ───────────────────────
    print(f"\n[Stage 2] Collecting OPD dataset from SFT model rollouts...")
    reseed("dataset")
    t0 = time.time()
    # Roll out the SFT model (π_ref) to collect trajectories
    sft_policy = lambda obs: sft_net.act(obs)
    sft_trajs = []
    for _ in range(args.n_offline):
        qid = int(np.random.randint(env.N_P))
        traj = rollout(env, sft_policy, person_id=qid)
        sft_trajs.append(traj)

    dataset = OfflineDataset(sft_trajs, env)
    print(f"  Dataset size: {len(dataset)} transitions")
    print(f"  Support states: {len(dataset.support_keys)}")
    print(f"  SFT rollout success rate: {np.mean([t.success for t in sft_trajs]):.2%}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Method 2: Online RL ───────────────────────────────────────────────────
    if "online_rl" in args.methods:
        print(f"\n[Method 2/5] Training Online RL (REINFORCE)...")
        reseed("online_rl")
        t0 = time.time()
        rl_net, rl_rewards = online_rl_train(
            env, n_episodes=args.rl_episodes, lr=args.lr,
            batch_size=args.rl_batch_size, gamma=args.gamma, verbose=True
        )
        rl_results = evaluate_policy(env, rl_net, n=args.n_eval)
        print(f"  Final reward (last 100): {np.mean(rl_rewards[-100:]):.3f}")
        print(f"  Eval success rate: {rl_results['success_rate']:.2%}")
        print(f"  Avg steps: {rl_results['avg_steps']:.2f}")
        print(f"  Time: {time.time()-t0:.1f}s")
        results["methods"]["online_rl"] = {
            "rewards": rl_rewards, **rl_results, "time": time.time()-t0
        }

    # ── Method 3: Offline OPD ─────────────────────────────────────────────────
    if "offline_opd" in args.methods:
        print(f"\n[Method 3/5] Training Offline OPD (Lightning OPD)...")
        reseed("offline_opd")
        t0 = time.time()
        opd_net, opd_losses, opd_off_support = offline_opd_train(
            env, dataset, sft_net, epochs=args.opd_epochs, lr=args.lr,
            batch_size=args.batch_size, adv_clip=args.adv_clip, verbose=True
        )
        opd_results = evaluate_policy(env, opd_net, n=args.n_eval)
        print(f"  Final loss: {opd_losses[-1]:.4f}")
        print(f"  Final off-support ratio: {opd_off_support[-1]:.3f}")
        print(f"  Eval success rate: {opd_results['success_rate']:.2%}")
        print(f"  Avg steps: {opd_results['avg_steps']:.2f}")
        print(f"  Time: {time.time()-t0:.1f}s")
        results["methods"]["offline_opd"] = {
            "losses": opd_losses, "off_support": opd_off_support,
            **opd_results, "time": time.time()-t0
        }

    # ── Method 4: Online OPD (upper bound) ────────────────────────────────────
    if "online_opd" in args.methods:
        print(f"\n[Method 4/5] Training Online OPD (upper bound)...")
        reseed("online_opd_m")
        t0 = time.time()
        online_opd_net, online_opd_rewards = online_opd_train(
            env, sft_net, n_episodes=args.rl_episodes, lr=args.lr,
            batch_size=args.rl_batch_size, gamma=args.gamma,
            adv_clip=args.adv_clip, verbose=True
        )
        online_opd_results = evaluate_policy(env, online_opd_net, n=args.n_eval)
        print(f"  Final reward (last 100): {np.mean(online_opd_rewards[-100:]):.3f}")
        print(f"  Eval success rate: {online_opd_results['success_rate']:.2%}")
        print(f"  Avg steps: {online_opd_results['avg_steps']:.2f}")
        print(f"  Time: {time.time()-t0:.1f}s")
        results["methods"]["online_opd"] = {
            "rewards": online_opd_rewards, **online_opd_results,
            "time": time.time()-t0
        }

    # ── Method 5: DAgger OPD ──────────────────────────────────────────────────
    if "dagger_opd" in args.methods:
        print(f"\n[Method 5/5] Training DAgger OPD (periodic refresh)...")
        reseed("dagger_opd")
        t0 = time.time()
        dagger_net, dagger_losses, dagger_off_support = dagger_opd_train(
            env, dataset, sft_net, epochs=args.opd_epochs, lr=args.lr,
            batch_size=args.batch_size, adv_clip=args.adv_clip,
            refresh_every=args.dagger_refresh_every,
            refresh_eps=args.dagger_refresh_eps, verbose=True
        )
        dagger_results = evaluate_policy(env, dagger_net, n=args.n_eval)
        print(f"  Final loss: {dagger_losses[-1]:.4f}")
        print(f"  Final off-support ratio: {dagger_off_support[-1]:.3f}")
        print(f"  Eval success rate: {dagger_results['success_rate']:.2%}")
        print(f"  Avg steps: {dagger_results['avg_steps']:.2f}")
        print(f"  Time: {time.time()-t0:.1f}s")
        results["methods"]["dagger_opd"] = {
            "losses": dagger_losses, "off_support": dagger_off_support,
            **dagger_results, "time": time.time()-t0
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Success Rate':<15} {'Avg Steps':<12} {'Time (s)':<10}")
    print("-" * 70)

    for method_name in args.methods:
        if method_name in results["methods"]:
            res = results["methods"][method_name]
            print(f"{method_name:<20} {res['success_rate']:>6.2%}         "
                  f"{res['avg_steps']:>6.2f}       {res['time']:>7.1f}")

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ── Plot if requested ─────────────────────────────────────────────────────
    if args.plot:
        from plot import plot_results
        plot_results(results, output_dir)
        print(f"Plots saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Lightning OPD toy experiment on multi-hop KG-QA"
    )

    # Environment
    parser.add_argument("--n_persons", type=int, default=20)
    parser.add_argument("--n_companies", type=int, default=20)
    parser.add_argument("--n_countries", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Data collection
    parser.add_argument("--n_offline", type=int, default=300,
                        help="Number of offline oracle trajectories")
    parser.add_argument("--oracle_eps", type=float, default=0.0,
                        help="Oracle epsilon (0=perfect, >0=noisy)")

    # Training
    parser.add_argument("--methods", nargs="+",
                        default=["sft", "online_rl", "offline_opd", "online_opd", "dagger_opd"],
                        choices=["sft", "online_rl", "offline_opd", "online_opd", "dagger_opd"])
    parser.add_argument("--sft_epochs", type=int, default=60)
    parser.add_argument("--opd_epochs", type=int, default=40)
    parser.add_argument("--rl_episodes", type=int, default=3000)
    parser.add_argument("--rl_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--adv_clip", type=float, default=2.0)

    # DAgger
    parser.add_argument("--dagger_refresh_every", type=int, default=8)
    parser.add_argument("--dagger_refresh_eps", type=int, default=80)

    # Evaluation
    parser.add_argument("--n_eval", type=int, default=500,
                        help="Number of episodes for final evaluation")

    # Output
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_experiment(args)


if __name__ == "__main__":
    main()
