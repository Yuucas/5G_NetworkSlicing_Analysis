"""
Test script to verify RL environment works correctly.
This tests the environment without full training.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import QoSDataLoader, DataConfig
from src.data.preprocessing import QoSPreprocessor

def main():
    print("=" * 60)
    print("Testing RL Environment Setup")
    print("=" * 60)

    # Load and prepare data
    print("\n1. Loading and preprocessing data...")
    data_path = "data/raw/Quality of Service 5G.csv"

    data_config = DataConfig(raw_data_path=data_path)
    loader = QoSDataLoader(data_config)
    loader.load_data()
    loader.validate_data()

    train_data, val_data, test_data = loader.split_data()
    print(f"   [OK] Data split: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    # Preprocess
    preprocessor = QoSPreprocessor(
        scaler_type="standard",
        encode_categorical=True,
        extract_time_features=True
    )

    train_processed = preprocessor.fit_transform(train_data)
    print(f"   [OK] Preprocessed: {train_processed.shape}")

    # Convert to numeric array
    numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'Resource_Allocation_Pct' in numeric_cols:
        numeric_cols.remove('Resource_Allocation_Pct')

    train_array = train_processed[numeric_cols].values
    print(f"   [OK] Numeric array: {train_array.shape}")
    print(f"   [OK] Features: {len(numeric_cols)}")

    # Show feature names
    print("\n2. Feature mapping:")
    for i, col in enumerate(numeric_cols[:15]):
        print(f"   [{i}] {col}")
    if len(numeric_cols) > 15:
        print(f"   ... and {len(numeric_cols) - 15} more")

    # Try to import and create environment
    print("\n3. Testing environment import...")
    try:
        from src.models.reinforcement_learning.environment import NetworkSlicingEnv
        print("   [OK] Environment module imported")

        # Create environment
        print("\n4. Creating environment...")
        env = NetworkSlicingEnv(
            data=train_array,
            max_bandwidth=1000.0,
            action_type="discrete",
            n_discrete_actions=5
        )
        print(f"   [OK] Environment created")
        print(f"   [OK] State space: {env.observation_space.shape}")
        print(f"   [OK] Action space: {env.action_space}")
        print(f"   [OK] Max steps: {env.max_steps}")

        # Test reset
        print("\n5. Testing environment reset...")
        state, info = env.reset()
        print(f"   [OK] Initial state shape: {state.shape}")
        print(f"   [OK] Initial state: {state}")
        print(f"   [OK] Info: {info}")

        # Test step
        print("\n6. Testing environment step...")
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"   [OK] Action: {action}")
        print(f"   [OK] Reward: {reward:.2f}")
        print(f"   [OK] Next state shape: {next_state.shape}")
        print(f"   [OK] Terminated: {terminated}")

        # Run a short episode
        print("\n7. Running short episode (10 steps)...")
        state, _ = env.reset()
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"   [INFO] Episode ended at step {i+1}")
                break

        print(f"   [OK] Episode completed")
        print(f"   [OK] Total reward: {total_reward:.2f}")
        print(f"   [OK] SLA violations: {info['sla_violations']}")
        print(f"   [OK] Network utilization: {info['network_utilization']:.2%}")

        print("\n" + "=" * 60)
        print("[OK] Environment test completed successfully!")
        print("=" * 60)
        print("\nNext: Install torch and gymnasium to train the agent:")
        print("  pip install torch gymnasium")
        print("  python scripts/train_rl_agent.py --episodes 10")

        return True

    except ImportError as e:
        print(f"   [INFO] Missing dependency: {e}")
        print("\n" + "=" * 60)
        print("[INFO] Environment requires additional packages")
        print("=" * 60)
        print("\nTo use the RL environment, install:")
        print("  pip install gymnasium numpy")
        print("\nTo train the DQN agent, install:")
        print("  pip install torch gymnasium stable-baselines3")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
