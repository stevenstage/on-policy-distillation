"""Plotting utilities for experimental results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def plot_results(results: Dict, output_dir: Path):
    """Generate comparison plots for all methods."""

    methods = results["methods"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Training curves ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves (SFT, Offline OPD, DAgger OPD)
    ax = axes[0]
    for method_name in ["sft", "offline_opd", "dagger_opd"]:
        if method_name in methods and "losses" in methods[method_name]:
            losses = methods[method_name]["losses"]
            ax.plot(losses, label=method_name.upper(), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Reward curves (Online RL, Online OPD)
    ax = axes[1]
    for method_name in ["online_rl", "online_opd"]:
        if method_name in methods and "rewards" in methods[method_name]:
            rewards = methods[method_name]["rewards"]
            # Smooth with moving average
            window = 50
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=method_name.replace("_", " ").upper(), linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Online Training Rewards")
    ax.legend()
    ax.grid(alpha=0.3)

    # Off-support ratio (Offline OPD, DAgger OPD)
    ax = axes[2]
    for method_name in ["offline_opd", "dagger_opd"]:
        if method_name in methods and "off_support" in methods[method_name]:
            off_support = methods[method_name]["off_support"]
            epochs = np.arange(5, len(off_support)*5 + 1, 5)
            ax.plot(epochs, off_support, marker='o',
                    label=method_name.replace("_", " ").upper(), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Off-Support Ratio")
    ax.set_title("Distribution Shift Tracking")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'training_curves.png'}")
    plt.close()

    # ── Figure 2: Final performance comparison ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    method_names = list(methods.keys())
    success_rates = [methods[m]["success_rate"] * 100 for m in method_names]
    avg_steps = [methods[m]["avg_steps"] for m in method_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))

    # Success rate
    ax = axes[0]
    bars = ax.bar(range(len(method_names)), success_rates, color=colors, alpha=0.8)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([m.replace("_", "\n").upper() for m in method_names],
                       rotation=0, ha='center')
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Final Evaluation: Success Rate")
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, success_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Average steps
    ax = axes[1]
    bars = ax.bar(range(len(method_names)), avg_steps, color=colors, alpha=0.8)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([m.replace("_", "\n").upper() for m in method_names],
                       rotation=0, ha='center')
    ax.set_ylabel("Average Steps")
    ax.set_title("Final Evaluation: Avg Steps to Answer")
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_steps)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "final_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'final_comparison.png'}")
    plt.close()

    # ── Figure 3: Method comparison table (text) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    table_data = [["Method", "Success Rate", "Avg Steps", "Avg Reward", "Time (s)"]]
    for m in method_names:
        row = [
            m.upper(),
            f"{methods[m]['success_rate']*100:.1f}%",
            f"{methods[m]['avg_steps']:.2f}",
            f"{methods[m]['avg_reward']:.3f}",
            f"{methods[m]['time']:.1f}",
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.savefig(output_dir / "results_table.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'results_table.png'}")
    plt.close()
