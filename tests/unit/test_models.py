"""
Unit tests for model modules.
"""

import pytest
import numpy as np

# Only import if dependencies available
try:
    import torch
    import gymnasium as gym
    from src.models.reinforcement_learning.environment import NetworkSlicingEnv
    from src.models.reinforcement_learning.dqn_agent import DQNAgent, DQNNetwork, ReplayBuffer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch and gymnasium")
class TestNetworkSlicingEnv:
    """Test NetworkSlicingEnv class."""

    def test_environment_initialization(self, sample_numeric_array):
        """Test environment initialization."""
        env = NetworkSlicingEnv(
            data=sample_numeric_array,
            max_bandwidth=1000.0,
            action_type="discrete",
            n_discrete_actions=5
        )

        assert env.max_steps == len(sample_numeric_array)
        assert env.action_space.n == 5
        assert env.observation_space.shape[0] == 10

    def test_environment_reset(self, sample_numeric_array):
        """Test environment reset."""
        env = NetworkSlicingEnv(data=sample_numeric_array)

        state, info = env.reset()

        assert isinstance(state, np.ndarray)
        assert state.shape == (10,)
        assert isinstance(info, dict)
        assert env.current_step == 0

    def test_environment_step(self, sample_numeric_array):
        """Test environment step."""
        env = NetworkSlicingEnv(data=sample_numeric_array, action_type="discrete")

        state, _ = env.reset()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_completion(self, sample_numeric_array):
        """Test that episode completes after all steps."""
        env = NetworkSlicingEnv(data=sample_numeric_array)

        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < len(sample_numeric_array) + 10:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done
        assert steps <= len(sample_numeric_array)

    def test_reward_calculation(self, sample_numeric_array):
        """Test that rewards are calculated."""
        env = NetworkSlicingEnv(data=sample_numeric_array)

        state, _ = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)

        # Check reward is numeric (can be int, float, or numpy numeric type)
        assert isinstance(reward, (int, float, np.number))
        assert not np.isnan(reward)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch")
class TestDQNNetwork:
    """Test DQNNetwork class."""

    def test_network_initialization(self):
        """Test network initialization."""
        network = DQNNetwork(state_dim=10, action_dim=5)

        assert isinstance(network, torch.nn.Module)

    def test_network_forward(self):
        """Test forward pass."""
        network = DQNNetwork(state_dim=10, action_dim=5)

        # Create sample input
        x = torch.randn(32, 10)  # Batch size 32, state dim 10
        output = network(x)

        assert output.shape == (32, 5)  # Batch size 32, action dim 5

    def test_network_custom_hidden_dims(self):
        """Test network with custom hidden dimensions."""
        network = DQNNetwork(state_dim=10, action_dim=5, hidden_dims=[128, 64])

        x = torch.randn(16, 10)
        output = network(x)

        assert output.shape == (16, 5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch")
class TestReplayBuffer:
    """Test ReplayBuffer class."""

    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=100)

        assert len(buffer) == 0

    def test_buffer_push(self):
        """Test adding transitions to buffer."""
        buffer = ReplayBuffer(capacity=100)

        state = np.random.randn(10)
        action = 1
        next_state = np.random.randn(10)
        reward = 1.0
        done = False

        buffer.push(state, action, next_state, reward, done)

        assert len(buffer) == 1

    def test_buffer_capacity(self):
        """Test that buffer respects capacity."""
        buffer = ReplayBuffer(capacity=10)

        # Add 20 transitions
        for i in range(20):
            buffer.push(np.random.randn(10), 0, np.random.randn(10), 0.0, False)

        assert len(buffer) == 10  # Should only keep last 10

    def test_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)

        # Add some transitions
        for i in range(50):
            buffer.push(np.random.randn(10), 0, np.random.randn(10), 0.0, False)

        # Sample batch
        batch = buffer.sample(batch_size=32)

        assert len(batch) == 32


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch")
class TestDQNAgent:
    """Test DQNAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(state_dim=10, action_dim=5)

        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.policy_net is not None
        assert agent.target_net is not None

    def test_agent_select_action(self):
        """Test action selection."""
        agent = DQNAgent(state_dim=10, action_dim=5)

        state = np.random.randn(10).astype(np.float32)

        # Test exploration (training=True)
        action = agent.select_action(state, training=True)
        assert isinstance(action, int)
        assert 0 <= action < 5

        # Test exploitation (training=False)
        action = agent.select_action(state, training=False)
        assert isinstance(action, int)
        assert 0 <= action < 5

    def test_agent_store_transition(self):
        """Test storing transitions."""
        agent = DQNAgent(state_dim=10, action_dim=5)

        state = np.random.randn(10)
        action = 1
        next_state = np.random.randn(10)
        reward = 1.0
        done = False

        agent.store_transition(state, action, next_state, reward, done)

        assert len(agent.memory) == 1

    def test_agent_train_step(self):
        """Test training step."""
        agent = DQNAgent(state_dim=10, action_dim=5, batch_size=32)

        # Fill buffer with enough samples
        for i in range(100):
            state = np.random.randn(10)
            action = np.random.randint(0, 5)
            next_state = np.random.randn(10)
            reward = np.random.randn()
            done = False
            agent.store_transition(state, action, next_state, reward, done)

        # Training step should return loss
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)

    def test_agent_epsilon_decay(self):
        """Test epsilon decay."""
        agent = DQNAgent(
            state_dim=10,
            action_dim=5,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=100
        )

        initial_epsilon = agent._get_epsilon()
        assert initial_epsilon == 1.0

        # Simulate many steps
        agent.steps_done = 1000
        final_epsilon = agent._get_epsilon()

        assert final_epsilon < initial_epsilon
        assert final_epsilon >= 0.01

    def test_agent_save_load(self, tmp_path):
        """Test saving and loading agent."""
        agent1 = DQNAgent(state_dim=10, action_dim=5)

        # Save
        save_path = tmp_path / "test_agent.pt"
        agent1.save(str(save_path))

        assert save_path.exists()

        # Load into new agent
        agent2 = DQNAgent(state_dim=10, action_dim=5)
        agent2.load(str(save_path))

        # Should have same number of steps
        assert agent2.steps_done == agent1.steps_done
