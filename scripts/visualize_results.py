"""
Visualize training results and model performance.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def plot_training_progress(checkpoint_path: str, output_dir: str = "results"):
    """
    Plot training progress from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        output_dir: Directory to save plots
    """
    print("=" * 60)
    print("Visualizing Training Progress")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("   [OK] Checkpoint loaded")
    except Exception as e:
        print(f"   [ERROR] Failed to load checkpoint: {e}")
        return

    # Extract metrics
    losses = checkpoint.get('losses', [])
    rewards = checkpoint.get('rewards', [])
    steps_done = checkpoint.get('steps_done', 0)

    print(f"\n2. Metrics found:")
    print(f"   - Training steps: {steps_done}")
    print(f"   - Loss values: {len(losses)}")
    print(f"   - Episode rewards: {len(rewards)}")

    if not losses and not rewards:
        print("\n   [WARNING] No training metrics found in checkpoint")
        print("   The checkpoint may have been saved before training")
        return

    # Create visualizations
    print(f"\n3. Creating visualizations...")

    # Figure 1: Training Loss
    if losses:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(losses, alpha=0.6, linewidth=0.5)
        # Moving average
        window = min(100, len(losses) // 10)
        if window > 1:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(losses)), moving_avg, 'r-', linewidth=2, label=f'{window}-step MA')
            plt.legend()
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(losses, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Loss Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        loss_plot_path = output_path / 'training_loss.png'
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {loss_plot_path}")
        plt.close()

    # Figure 2: Episode Rewards
    if rewards:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Raw rewards
        axes[0, 0].plot(rewards, alpha=0.6, linewidth=0.8)
        # Moving average
        window = min(10, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-episode MA')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Rewards', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Reward distribution
        axes[0, 1].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(rewards), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        axes[0, 1].legend()
        axes[0, 1].set_title('Reward Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Cumulative rewards
        cumulative_rewards = np.cumsum(rewards)
        axes[1, 0].plot(cumulative_rewards, linewidth=2)
        axes[1, 0].set_title('Cumulative Rewards', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].grid(True, alpha=0.3)

        # Reward improvement
        if len(rewards) > 10:
            # Compare first 10% vs last 10%
            split = len(rewards) // 10
            early_rewards = rewards[:split]
            late_rewards = rewards[-split:]

            axes[1, 1].boxplot([early_rewards, late_rewards], labels=['Early', 'Late'])
            axes[1, 1].set_title('Early vs Late Training Performance', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].grid(True, alpha=0.3)

            # Add statistics
            improvement = np.mean(late_rewards) - np.mean(early_rewards)
            axes[1, 1].text(0.5, 0.95, f'Improvement: {improvement:+.2f}',
                          transform=axes[1, 1].transAxes,
                          ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        reward_plot_path = output_path / 'episode_rewards.png'
        plt.savefig(reward_plot_path, dpi=150, bbox_inches='tight')
        print(f"   [OK] Saved: {reward_plot_path}")
        plt.close()

    # Figure 3: Training Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    summary_text = f"""
    TRAINING SUMMARY
    {'=' * 50}

    Training Steps:        {steps_done:,}
    Episodes:              {len(rewards) if rewards else 0:,}

    REWARDS
    Mean:                  {np.mean(rewards) if rewards else 0:.2f}
    Std:                   {np.std(rewards) if rewards else 0:.2f}
    Min:                   {np.min(rewards) if rewards else 0:.2f}
    Max:                   {np.max(rewards) if rewards else 0:.2f}

    LOSS
    Mean:                  {np.mean(losses) if losses else 0:.4f}
    Final:                 {losses[-1] if losses else 0:.4f}
    Min:                   {np.min(losses) if losses else 0:.4f}

    IMPROVEMENT (First 10% vs Last 10%)
    Early Mean:            {np.mean(rewards[:len(rewards)//10]) if len(rewards) > 10 else 0:.2f}
    Late Mean:             {np.mean(rewards[-len(rewards)//10:]) if len(rewards) > 10 else 0:.2f}
    Change:                {np.mean(rewards[-len(rewards)//10:]) - np.mean(rewards[:len(rewards)//10]) if len(rewards) > 10 else 0:+.2f}
    """

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    summary_plot_path = output_path / 'training_summary.png'
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    print(f"   [OK] Saved: {summary_plot_path}")
    plt.close()

    print("\n" + "=" * 60)
    print(f"[OK] All plots saved to: {output_dir}/")
    print("=" * 60)
    print("\nGenerated files:")
    for file in output_path.glob('*.png'):
        print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dqn_agent_best.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    plot_training_progress(args.checkpoint, args.output)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
