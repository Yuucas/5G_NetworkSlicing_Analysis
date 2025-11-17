"""
Base model class for all ML/DL models.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.metrics_history: Dict[str, list] = {}

    @abstractmethod
    def build(self) -> None:
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional training arguments

        Returns:
            Training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Evaluation metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "is_trained": self.is_trained,
        }

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log training/evaluation metrics."""
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(value)

        logger.info(f"Step {step}: {metrics}")
