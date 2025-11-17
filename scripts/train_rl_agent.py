"""
Training script for Reinforcement Learning agents.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import yaml
from datetime import datetime

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


def load_config(config_path: str = "config/model_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(data_path: str):
    """Load and preprocess data."""
    logger.info("Loading data...")

    # Load data
    data_config = DataConfig(raw_data_path=data_path)
    loader = QoSDataLoader(data_config)
    loader.load_data()
    loader.validate_data()

    # Split data
    train_data, val_data, test_data = loader.split_data()

    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = QoSPreprocessor(
        scaler_type="standard",
        encode_categorical=True,
        extract_time_features=True
    )

    train_processed = preprocessor.fit_transform(train_data)
    val_processed = preprocessor.transform(val_data)

    # Convert to numeric arrays for RL environment
    # Drop non-numeric columns (Timestamp, User_ID)
    numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target variable if present
    if 'Resource_Allocation_Pct' in numeric_cols:
        numeric_cols.remove('Resource_Allocation_Pct')

    train_array = train_processed[numeric_cols].values
    val_array = val_processed[numeric_cols].values

    logger.info(f"Prepared data - Train shape: {train_array.shape}, Val shape: {val_array.shape}")
    logger.info(f"Features used: {len(numeric_cols)} numeric features")

    return train_array, val_array, preprocessor


def train_dqn_agent(
    train_data: np.ndarray,
    val_data: np.ndarray,
    config: dict,
    episodes: int = 1000,
    checkpoint_dir: str = "checkpoints"
):
    """Train DQN agent."""
    logger.info("Training DQN agent...")

    # Create environment
    env = NetworkSlicingEnv(
        data=train_data if isinstance(train_data, np.ndarray) else train_data.values,
        **config['reinforcement_learning']['environment']
    )

    # Get actual state dimension from environment
    actual_state_dim = env.observation_space.shape[0]
    logger.info(f"State dimension: {actual_state_dim}")

    # Create agent with actual state dimension
    rl_config = config['reinforcement_learning']['dqn']
    agent = DQNAgent(
        state_dim=actual_state_dim,  # Use actual state dimension
        action_dim=rl_config['action_dim'],
        learning_rate=rl_config['learning_rate'],
        gamma=rl_config['gamma'],
        epsilon_start=rl_config['epsilon_start'],
        epsilon_end=rl_config['epsilon_end'],
        epsilon_decay=rl_config['epsilon_decay'],
        batch_size=rl_config['batch_size'],
        buffer_capacity=rl_config['buffer_capacity'],
        target_update_freq=rl_config['target_update_freq']
    )

    # Training loop
    best_reward = -np.inf
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, next_state, reward, done)

            # Train
            loss = agent.train_step()

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Epsilon: {agent._get_epsilon():.3f}"
            )

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{checkpoint_dir}/dqn_agent_best.pt")
                logger.info(f"Saved best model with reward: {best_reward:.2f}")

        # Save checkpoint
        if (episode + 1) % 100 == 0:
            agent.save(f"{checkpoint_dir}/dqn_agent_ep{episode+1}.pt")

    logger.info("Training complete!")
    return agent, episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Train RL Agent for 5G Network Slicing")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["dqn", "ppo", "a3c"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/Quality of Service 5G.csv",
        help="Path to dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Prepare data
    train_data, val_data, preprocessor = prepare_data(args.data)

    # Train agent
    if args.algorithm == "dqn":
        agent, rewards = train_dqn_agent(
            train_data,
            val_data,
            config,
            episodes=args.episodes,
            checkpoint_dir=args.checkpoint_dir
        )
    else:
        logger.error(f"Algorithm {args.algorithm} not implemented yet")
        return

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
