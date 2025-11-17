"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_qos_data():
    """Create sample QoS data for testing."""
    data = {
        "Timestamp": ["9/3/2023 10:00"] * 10,
        "User_ID": [f"User_{i}" for i in range(10)],
        "Application_Type": [
            "Video_Call",
            "Voice_Call",
            "Streaming",
            "Emergency_Service",
            "Online_Gaming",
        ]
        * 2,
        "Signal_Strength": ["-75 dBm", "-80 dBm", "-85 dBm", "-70 dBm", "-78 dBm"] * 2,
        "Latency": ["30 ms", "20 ms", "40 ms", "10 ms", "25 ms"] * 2,
        "Required_Bandwidth": ["10 Mbps", "100 Kbps", "5 Mbps", "1 Mbps", "2 Mbps"] * 2,
        "Allocated_Bandwidth": ["15 Mbps", "120 Kbps", "6 Mbps", "1.5 Mbps", "3 Mbps"] * 2,
        "Resource_Allocation": ["70%", "80%", "75%", "90%", "85%"] * 2,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_qos_data):
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "test_data.csv"
    sample_qos_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def sample_numeric_array():
    """Create sample numeric array for RL environment."""
    # 10 samples, 10 features
    return np.random.randn(10, 10).astype(np.float32)


@pytest.fixture
def processed_data():
    """Create sample preprocessed data."""
    data = {
        "Signal_Strength_dBm": np.random.uniform(-100, -50, 20),
        "Signal_Quality": np.random.uniform(0, 1, 20),
        "Latency_ms": np.random.uniform(10, 100, 20),
        "Required_Bandwidth_Mbps": np.random.uniform(0.1, 20, 20),
        "Allocated_Bandwidth_Mbps": np.random.uniform(0.1, 25, 20),
        "Bandwidth_Utilization": np.random.uniform(0.5, 1.5, 20),
        "Is_Overallocated": np.random.randint(0, 2, 20),
        "Resource_Allocation_Pct": np.random.uniform(50, 100, 20),
        "Hour": np.random.randint(0, 24, 20),
        "DayOfWeek": np.random.randint(0, 7, 20),
        "DayOfMonth": np.random.randint(1, 32, 20),
        "Hour_Sin": np.random.uniform(-1, 1, 20),
        "Hour_Cos": np.random.uniform(-1, 1, 20),
        "DayOfWeek_Sin": np.random.uniform(-1, 1, 20),
        "DayOfWeek_Cos": np.random.uniform(-1, 1, 20),
        "Is_Peak_Hour": np.random.randint(0, 2, 20),
        "Application_Type_Encoded": np.random.randint(0, 11, 20),
        "User_ID_Encoded": np.random.randint(0, 100, 20),
        "Timestamp": pd.date_range("2023-09-03 10:00", periods=20, freq="T"),
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "reinforcement_learning": {
            "dqn": {
                "state_dim": 10,
                "action_dim": 5,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 1000,
                "batch_size": 32,
                "buffer_capacity": 1000,
                "target_update_freq": 100,
            },
            "environment": {
                "max_bandwidth": 1000.0,
                "action_type": "discrete",
                "n_discrete_actions": 5,
                "sla_violation_penalty": -10.0,
                "overallocation_penalty": -5.0,
                "efficiency_reward": 5.0,
                "emergency_priority_bonus": 10.0,
            },
        }
    }
