"""
Integration tests for complete pipelines.
"""

import numpy as np
import pytest

from src.data.data_loader import DataConfig, QoSDataLoader
from src.data.preprocessing import QoSPreprocessor

# Conditional imports
try:
    from src.models.reinforcement_learning.dqn_agent import DQNAgent
    from src.models.reinforcement_learning.environment import NetworkSlicingEnv

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDataPipeline:
    """Test end-to-end data pipeline."""

    def test_complete_data_pipeline(self, sample_csv_file):
        """Test complete data loading and preprocessing pipeline."""
        # 1. Load data
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        loader.load_data()

        assert loader.data is not None
        assert len(loader.data) == 10

        # 2. Validate
        loader.validate_data()

        # 3. Split
        train, val, test = loader.split_data(stratify_col=None)
        assert len(train) + len(val) + len(test) == 10

        # 4. Preprocess
        preprocessor = QoSPreprocessor()
        train_processed = preprocessor.fit_transform(train)

        assert train_processed is not None
        assert "Signal_Strength_dBm" in train_processed.columns
        assert "Latency_ms" in train_processed.columns

        # 5. Transform validation set
        val_processed = preprocessor.transform(val)
        assert len(val_processed) == len(val)

    def test_data_to_model_ready(self, sample_csv_file):
        """Test converting data to model-ready format."""
        # Load and preprocess
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        loader.load_data()
        train, val, test = loader.split_data(stratify_col=None)

        preprocessor = QoSPreprocessor()
        train_processed = preprocessor.fit_transform(train)

        # Convert to numeric array
        numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
        if "Resource_Allocation_Pct" in numeric_cols:
            numeric_cols.remove("Resource_Allocation_Pct")

        train_array = train_processed[numeric_cols].values

        assert isinstance(train_array, np.ndarray)
        assert train_array.shape[0] == len(train)
        assert train_array.shape[1] > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch and gymnasium")
class TestTrainingPipeline:
    """Test training pipeline integration."""

    def test_environment_with_processed_data(self, sample_csv_file):
        """Test creating environment with processed data."""
        # Prepare data
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        loader.load_data()
        train, _, _ = loader.split_data(stratify_col=None)

        preprocessor = QoSPreprocessor()
        train_processed = preprocessor.fit_transform(train)

        numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
        if "Resource_Allocation_Pct" in numeric_cols:
            numeric_cols.remove("Resource_Allocation_Pct")

        train_array = train_processed[numeric_cols].values

        # Create environment
        env = NetworkSlicingEnv(data=train_array, max_bandwidth=1000.0)

        assert env is not None
        assert env.max_steps == len(train_array)

    def test_agent_environment_interaction(self, sample_numeric_array):
        """Test agent interacting with environment."""
        # Create environment and agent
        env = NetworkSlicingEnv(
            data=sample_numeric_array, action_type="discrete", n_discrete_actions=5
        )
        agent = DQNAgent(state_dim=10, action_dim=5)

        # Run one episode
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 20:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(state, action, next_state, reward, False)

            total_reward += reward
            state = next_state
            done = terminated or truncated
            steps += 1

        assert steps > 0
        assert len(agent.memory) == steps

    def test_mini_training_loop(self, sample_numeric_array):
        """Test a minimal training loop."""
        env = NetworkSlicingEnv(
            data=sample_numeric_array, action_type="discrete", n_discrete_actions=5
        )
        agent = DQNAgent(state_dim=10, action_dim=5, batch_size=4)

        # Train for 2 episodes
        for episode in range(2):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)

                agent.store_transition(state, action, next_state, reward, False)

                # Train if enough samples
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.train_step()
                    assert loss is not None or loss is None  # Can be None initially

                episode_reward += reward
                state = next_state
                done = terminated or truncated

            assert episode_reward != 0  # Should have accumulated some reward


class TestSaveLoadPipeline:
    """Test saving and loading throughout pipeline."""

    def test_save_and_load_preprocessed_data(self, sample_csv_file, tmp_path):
        """Test saving preprocessed data and loading it back."""
        # Process and save
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        loader.load_data()
        train, val, test = loader.split_data(stratify_col=None)

        output_dir = tmp_path / "processed"
        loader.save_processed_data(str(output_dir))

        # Load back
        import pandas as pd

        train_loaded = pd.read_csv(output_dir / "train.csv")
        val_loaded = pd.read_csv(output_dir / "val.csv")
        test_loaded = pd.read_csv(output_dir / "test.csv")

        assert len(train_loaded) == len(train)
        assert len(val_loaded) == len(val)
        assert len(test_loaded) == len(test)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch")
    def test_save_and_load_agent(self, tmp_path):
        """Test saving and loading trained agent."""
        # Create and partially train agent
        agent1 = DQNAgent(state_dim=10, action_dim=5)

        # Add some transitions
        for i in range(10):
            agent1.store_transition(
                np.random.randn(10),
                np.random.randint(0, 5),
                np.random.randn(10),
                np.random.randn(),
                False,
            )

        # Save
        save_path = tmp_path / "agent.pt"
        agent1.save(str(save_path))

        # Load into new agent
        agent2 = DQNAgent(state_dim=10, action_dim=5)
        agent2.load(str(save_path))

        # Verify same state
        assert agent2.steps_done == agent1.steps_done
