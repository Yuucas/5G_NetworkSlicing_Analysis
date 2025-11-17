"""
Evaluate trained DQN agent on test data.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import argparse
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import QoSDataLoader, DataConfig
from src.data.preprocessing import QoSPreprocessor
from src.models.reinforcement_learning.environment import NetworkSlicingEnv
from src.models.reinforcement_learning.dqn_agent import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_agent(
    agent_path: str,
    data_path: str,
    num_episodes: int = 10,
    max_bandwidth: float = 1000.0
):
    """
    Evaluate trained agent on test data.

    Args:
        agent_path: Path to saved agent checkpoint
        data_path: Path to dataset
        num_episodes: Number of episodes to evaluate
        max_bandwidth: Maximum network bandwidth
    """
    print("=" * 60)
    print("Evaluating Trained DQN Agent")
    print("=" * 60)

    # Load and prepare data
    print("\n1. Loading test data...")
    data_config = DataConfig(raw_data_path=data_path)
    loader = QoSDataLoader(data_config)
    loader.load_data()
    train_data, val_data, test_data = loader.split_data()

    print(f"   [OK] Test data: {len(test_data)} samples")

    # Preprocess
    print("\n2. Preprocessing data...")
    preprocessor = QoSPreprocessor(
        scaler_type="standard",
        encode_categorical=True,
        extract_time_features=True
    )

    preprocessor.fit_transform(train_data)
    test_processed = preprocessor.transform(test_data)

    # Convert to numeric array
    numeric_cols = test_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'Resource_Allocation_Pct' in numeric_cols:
        numeric_cols.remove('Resource_Allocation_Pct')

    test_array = test_processed[numeric_cols].values
    print(f"   [OK] Test array shape: {test_array.shape}")

    # Create environment
    print("\n3. Creating environment...")
    env = NetworkSlicingEnv(
        data=test_array,
        max_bandwidth=max_bandwidth,
        action_type="discrete",
        n_discrete_actions=5
    )
    print(f"   [OK] Environment created")

    # Load agent
    print(f"\n4. Loading trained agent from: {agent_path}")
    state_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_dim=state_dim, action_dim=5)

    try:
        agent.load(agent_path)
        print(f"   [OK] Agent loaded successfully")
    except Exception as e:
        print(f"   [ERROR] Failed to load agent: {e}")
        return

    # Evaluate
    print(f"\n5. Running evaluation ({num_episodes} episodes)...")
    print("-" * 60)

    episode_rewards = []
    sla_violations_list = []
    network_utilizations = []
    efficiency_scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            # Select action (greedy, no exploration)
            action = agent.select_action(state, training=False)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state
            steps += 1

        episode_rewards.append(episode_reward)
        sla_violations_list.append(info['sla_violations'])
        network_utilizations.append(info['network_utilization'])
        efficiency_scores.append(info['avg_efficiency'])

        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Reward: {episode_reward:8.2f} | Steps: {steps:3d} | "
              f"SLA Violations: {info['sla_violations']:3d} | "
              f"Utilization: {info['network_utilization']:.1%} | "
              f"Efficiency: {info['avg_efficiency']:.1%}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nReward Statistics:")
    print(f"  Mean:   {np.mean(episode_rewards):8.2f}")
    print(f"  Std:    {np.std(episode_rewards):8.2f}")
    print(f"  Min:    {np.min(episode_rewards):8.2f}")
    print(f"  Max:    {np.max(episode_rewards):8.2f}")

    print(f"\nSLA Violations:")
    print(f"  Mean:   {np.mean(sla_violations_list):8.2f}")
    print(f"  Total:  {np.sum(sla_violations_list):8.0f}")

    print(f"\nNetwork Utilization:")
    print(f"  Mean:   {np.mean(network_utilizations):8.1%}")
    print(f"  Std:    {np.std(network_utilizations):8.1%}")

    print(f"\nResource Efficiency:")
    print(f"  Mean:   {np.mean(efficiency_scores):8.1%}")
    print(f"  Std:    {np.std(efficiency_scores):8.1%}")

    # Compare with random baseline
    print("\n" + "-" * 60)
    print("RANDOM BASELINE COMPARISON")
    print("-" * 60)

    random_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()  # Random action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        random_rewards.append(episode_reward)

    print(f"\nRandom Policy Reward: {np.mean(random_rewards):8.2f} ± {np.std(random_rewards):6.2f}")
    print(f"Trained Agent Reward: {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):6.2f}")
    improvement = np.mean(episode_rewards) - np.mean(random_rewards)
    print(f"Improvement:          {improvement:8.2f} ({improvement/abs(np.mean(random_rewards))*100:+.1f}%)")

    print("\n" + "=" * 60)
    print("[OK] Evaluation completed!")
    print("=" * 60)

    return {
        'episode_rewards': episode_rewards,
        'sla_violations': sla_violations_list,
        'network_utilizations': network_utilizations,
        'efficiency_scores': efficiency_scores,
        'random_rewards': random_rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dqn_agent_best.pt",
        help="Path to agent checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/Quality of Service 5G.csv",
        help="Path to dataset"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )

    args = parser.parse_args()

    results = evaluate_agent(
        agent_path=args.checkpoint,
        data_path=args.data,
        num_episodes=args.episodes
    )

    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
