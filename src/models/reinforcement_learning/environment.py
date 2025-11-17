"""
Custom Gym environment for 5G Network Resource Allocation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class NetworkSlicingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for 5G Network Slicing.

    State Space:
        - Signal Strength (normalized)
        - Latency (ms)
        - Required Bandwidth (Mbps)
        - Current Allocated Bandwidth (Mbps)
        - Application Type (encoded)
        - Time features (hour, day of week)
        - Network load (current total allocation)
        - Historical allocation efficiency

    Action Space:
        - Continuous: Resource allocation percentage [0, 1]
        OR
        - Discrete: Allocation levels (e.g., 0%, 25%, 50%, 75%, 100%)

    Reward:
        - Positive: Meeting QoS requirements
        - Negative: SLA violations, overallocation, underutilization
        - Bonus: Efficient resource usage, priority handling (emergency services)
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        data: np.ndarray,
        max_bandwidth: float = 1000.0,  # Total available bandwidth in Mbps
        action_type: str = "continuous",  # "continuous" or "discrete"
        n_discrete_actions: int = 5,
        sla_violation_penalty: float = -10.0,
        overallocation_penalty: float = -5.0,
        efficiency_reward: float = 5.0,
        emergency_priority_bonus: float = 10.0,
    ):
        super().__init__()

        self.data = data
        self.max_bandwidth = max_bandwidth
        self.action_type = action_type
        self.sla_violation_penalty = sla_violation_penalty
        self.overallocation_penalty = overallocation_penalty
        self.efficiency_reward = efficiency_reward
        self.emergency_priority_bonus = emergency_priority_bonus

        # State space dimension
        # [signal_quality, latency, required_bw, allocated_bw, app_type,
        #  hour_sin, hour_cos, dow_sin, dow_cos, network_load]
        self.state_dim = 10

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Action space
        if action_type == "continuous":
            # Continuous allocation percentage [0, 1]
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        elif action_type == "discrete":
            # Discrete allocation levels
            self.action_space = spaces.Discrete(n_discrete_actions)
            self.discrete_levels = np.linspace(0, 1, n_discrete_actions)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # Episode tracking
        self.current_step = 0
        self.max_steps = len(data)
        self.current_state = None
        self.total_allocated = 0.0
        self.episode_reward = 0.0

        # Metrics
        self.sla_violations = 0
        self.overallocations = 0
        self.efficiency_scores = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.total_allocated = 0.0
        self.episode_reward = 0.0
        self.sla_violations = 0
        self.overallocations = 0
        self.efficiency_scores = []

        self.current_state = self._get_state()

        info = self._get_info()

        return self.current_state, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Resource allocation action

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to allocation percentage
        if self.action_type == "discrete":
            allocation_pct = self.discrete_levels[action]
        else:
            allocation_pct = np.clip(action[0], 0.0, 1.0)

        # Get current request details
        request = self._get_current_request()
        required_bw = request["required_bandwidth"]
        app_type = request["application_type"]
        latency = request["latency"]

        # Calculate allocated bandwidth
        allocated_bw = allocation_pct * required_bw * 1.5  # Allow some overallocation

        # Calculate reward
        reward = self._calculate_reward(
            required_bw=required_bw,
            allocated_bw=allocated_bw,
            app_type=app_type,
            latency=latency,
            allocation_pct=allocation_pct
        )

        self.episode_reward += reward
        self.total_allocated += allocated_bw

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get next state
        if not terminated:
            self.current_state = self._get_state()
        else:
            self.current_state = np.zeros(self.state_dim, dtype=np.float32)

        info = self._get_info()

        return self.current_state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        if self.current_step >= self.max_steps:
            return np.zeros(self.state_dim, dtype=np.float32)

        request = self._get_current_request()

        state = np.array([
            request["signal_quality"],
            request["latency"],
            request["required_bandwidth"],
            request.get("current_allocated", 0.0),
            request["application_type"],
            request["hour_sin"],
            request["hour_cos"],
            request["dow_sin"],
            request["dow_cos"],
            self.total_allocated / self.max_bandwidth,  # Network load
        ], dtype=np.float32)

        return state

    def _get_current_request(self) -> Dict[str, float]:
        """Get current network request details."""
        if self.current_step >= len(self.data):
            return {
                "signal_quality": 0.0,
                "latency": 0.0,
                "required_bandwidth": 0.0,
                "application_type": 0.0,
                "hour_sin": 0.0,
                "hour_cos": 0.0,
                "dow_sin": 0.0,
                "dow_cos": 0.0,
            }

        # Parse current data row
        # Preprocessed data is a numpy array with numeric features
        row = self.data[self.current_step]

        # Convert to float to ensure compatibility
        def safe_float(val, default=0.0):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        # Map indices to features based on preprocessing output
        # Expected order: Signal_Quality, Latency_ms, Required_Bandwidth_Mbps,
        # Allocated_Bandwidth_Mbps, Bandwidth_Utilization, Is_Overallocated,
        # Hour, DayOfWeek, DayOfMonth, Hour_Sin, Hour_Cos, DayOfWeek_Sin, DayOfWeek_Cos, etc.

        return {
            "signal_quality": safe_float(row[0] if len(row) > 0 else 0.0),
            "latency": safe_float(row[1] if len(row) > 1 else 0.0),
            "required_bandwidth": safe_float(row[2] if len(row) > 2 else 0.0),
            "current_allocated": safe_float(row[3] if len(row) > 3 else 0.0),
            "application_type": safe_float(row[4] if len(row) > 4 else 0.0),
            "hour_sin": safe_float(row[9] if len(row) > 9 else 0.0),
            "hour_cos": safe_float(row[10] if len(row) > 10 else 0.0),
            "dow_sin": safe_float(row[11] if len(row) > 11 else 0.0),
            "dow_cos": safe_float(row[12] if len(row) > 12 else 0.0),
        }

    def _calculate_reward(
        self,
        required_bw: float,
        allocated_bw: float,
        app_type: int,
        latency: float,
        allocation_pct: float
    ) -> float:
        """
        Calculate reward for current allocation decision.

        Reward components:
        1. QoS satisfaction: positive if allocation meets requirements
        2. Efficiency: penalize overallocation
        3. Priority: bonus for emergency services
        4. Network health: penalize excessive total allocation
        """
        reward = 0.0

        # 1. QoS satisfaction
        if allocated_bw >= required_bw:
            reward += self.efficiency_reward
        else:
            # SLA violation
            violation_ratio = (required_bw - allocated_bw) / required_bw
            reward += self.sla_violation_penalty * violation_ratio
            self.sla_violations += 1

        # 2. Efficiency penalty for overallocation
        if allocated_bw > required_bw * 1.2:  # More than 20% overallocation
            overalloc_ratio = (allocated_bw - required_bw) / required_bw
            reward += self.overallocation_penalty * overalloc_ratio
            self.overallocations += 1

        # 3. Emergency service priority bonus
        # Assuming app_type encoding where emergency services have specific codes
        if app_type in [3, 4] and allocated_bw >= required_bw:  # Emergency services
            reward += self.emergency_priority_bonus

        # 4. Network congestion penalty
        if self.total_allocated > self.max_bandwidth * 0.9:
            congestion_penalty = -2.0 * (self.total_allocated / self.max_bandwidth - 0.9)
            reward += congestion_penalty

        # 5. Efficiency score tracking
        efficiency = min(allocated_bw / required_bw, 1.0) if required_bw > 0 else 0.0
        self.efficiency_scores.append(efficiency)

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {
            "step": self.current_step,
            "total_allocated": self.total_allocated,
            "network_utilization": self.total_allocated / self.max_bandwidth,
            "sla_violations": self.sla_violations,
            "overallocations": self.overallocations,
            "avg_efficiency": np.mean(self.efficiency_scores) if self.efficiency_scores else 0.0,
            "episode_reward": self.episode_reward,
        }

    def render(self) -> None:
        """Render the environment (optional)."""
        if self.current_step < len(self.data):
            info = self._get_info()
            print(f"\nStep: {info['step']}")
            print(f"Network Utilization: {info['network_utilization']:.2%}")
            print(f"SLA Violations: {info['sla_violations']}")
            print(f"Episode Reward: {info['episode_reward']:.2f}")

    def close(self) -> None:
        """Clean up resources."""
        pass
