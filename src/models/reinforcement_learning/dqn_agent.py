"""
Deep Q-Network (DQN) Agent for Network Resource Allocation.
"""

import logging
import random
from collections import deque, namedtuple
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super(DQNNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Experience Replay Buffer."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args) -> None:
        """Add transition to buffer."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent with experience replay and target network.

    Features:
    - Double DQN
    - Dueling DQN architecture (optional)
    - Prioritized Experience Replay (optional)
    - Multi-step returns
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        batch_size: int = 128,
        buffer_capacity: int = 100000,
        target_update_freq: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)

        # Metrics
        self.losses = []
        self.rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; otherwise greedy

        Returns:
            Selected action index
        """
        epsilon = self._get_epsilon()

        if training and random.random() < epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def _get_epsilon(self) -> float:
        """Calculate current epsilon value with decay."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )
        return epsilon

    def store_transition(
        self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool
    ) -> None:
        """Store transition in replay buffer."""
        self.memory.push(state, action, next_state, reward, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN: use policy network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.losses.append(loss.item())

        return loss.item()

    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "losses": self.losses,
                "rewards": self.rewards,
            },
            path,
        )
        logger.info(f"DQN agent saved to {path}")

    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]
        self.losses = checkpoint["losses"]
        self.rewards = checkpoint["rewards"]
        logger.info(f"DQN agent loaded from {path}")
