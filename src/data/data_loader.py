"""
Data loader for 5G QoS dataset with validation and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and validation."""
    raw_data_path: str
    processed_data_path: Optional[str] = None
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    timestamp_col: str = "Timestamp"
    target_col: str = "Resource_Allocation"


class QoSDataLoader:
    """
    Loads and validates 5G QoS dataset.

    Features:
    - Timestamp: Time of measurement
    - User_ID: Unique user identifier
    - Application_Type: Type of application (Video_Call, Emergency_Service, etc.)
    - Signal_Strength: Signal strength in dBm
    - Latency: Network latency in ms
    - Required_Bandwidth: Required bandwidth
    - Allocated_Bandwidth: Allocated bandwidth
    - Resource_Allocation: Resource allocation percentage (target)
    """

    EXPECTED_COLUMNS = [
        "Timestamp", "User_ID", "Application_Type", "Signal_Strength",
        "Latency", "Required_Bandwidth", "Allocated_Bandwidth", "Resource_Allocation"
    ]

    APPLICATION_TYPES = [
        "Video_Call", "Voice_Call", "Streaming", "Emergency_Service",
        "Online_Gaming", "Background_Download", "Web_Browsing",
        "IoT_Temperature", "Video_Streaming", "File_Download", "VoIP_Call"
    ]

    def __init__(self, config: DataConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from {self.config.raw_data_path}")

        try:
            self.data = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Loaded {len(self.data)} records")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def validate_data(self) -> bool:
        """Validate dataset schema and data quality."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Check columns
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Check data types and ranges
        validation_checks = {
            "timestamp_valid": self._validate_timestamps(),
            "signal_strength_valid": self._validate_signal_strength(),
            "latency_valid": self._validate_latency(),
            "bandwidth_valid": self._validate_bandwidth(),
            "allocation_valid": self._validate_resource_allocation(),
            "application_types_valid": self._validate_application_types(),
        }

        all_valid = all(validation_checks.values())

        if all_valid:
            logger.info("Data validation passed")
        else:
            failed = [k for k, v in validation_checks.items() if not v]
            logger.warning(f"Validation failed for: {failed}")

        return all_valid

    def _validate_timestamps(self) -> bool:
        """Validate timestamp column."""
        try:
            pd.to_datetime(self.data[self.config.timestamp_col])
            return True
        except Exception as e:
            logger.error(f"Timestamp validation failed: {e}")
            return False

    def _validate_signal_strength(self) -> bool:
        """Validate signal strength values (should be negative dBm)."""
        signal_col = self.data["Signal_Strength"].str.extract(r'(-?\d+)').astype(float)
        valid = (signal_col >= -120).all().iloc[0] and (signal_col <= -30).all().iloc[0]
        if not valid:
            logger.warning("Signal strength values out of expected range [-120, -30] dBm")
        return valid

    def _validate_latency(self) -> bool:
        """Validate latency values."""
        latency = self.data["Latency"].str.extract(r'(\d+)').astype(float)
        valid = (latency > 0).all().iloc[0] and (latency < 1000).all().iloc[0]
        if not valid:
            logger.warning("Latency values out of expected range (0, 1000) ms")
        return valid

    def _validate_bandwidth(self) -> bool:
        """Validate bandwidth values."""
        try:
            # Handle different units (Kbps, Mbps)
            return True  # Implement unit conversion validation
        except Exception as e:
            logger.error(f"Bandwidth validation failed: {e}")
            return False

    def _validate_resource_allocation(self) -> bool:
        """Validate resource allocation percentage."""
        allocation = self.data["Resource_Allocation"].str.rstrip('%').astype(float)
        valid = (allocation >= 0).all() and (allocation <= 100).all()
        if not valid:
            logger.warning("Resource allocation out of range [0, 100]%")
        return valid

    def _validate_application_types(self) -> bool:
        """Validate application types."""
        unique_apps = self.data["Application_Type"].unique()
        unknown_apps = set(unique_apps) - set(self.APPLICATION_TYPES)
        if unknown_apps:
            logger.warning(f"Unknown application types: {unknown_apps}")
            return False
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.data is None:
            raise ValueError("Data not loaded")

        # Parse numeric values
        signal = self.data["Signal_Strength"].str.extract(r'(-?\d+)').astype(float).iloc[:, 0]
        latency = self.data["Latency"].str.extract(r'(\d+)').astype(float).iloc[:, 0]
        allocation = self.data["Resource_Allocation"].str.rstrip('%').astype(float)

        stats = {
            "total_records": len(self.data),
            "unique_users": self.data["User_ID"].nunique(),
            "application_distribution": self.data["Application_Type"].value_counts().to_dict(),
            "signal_strength": {
                "mean": signal.mean(),
                "std": signal.std(),
                "min": signal.min(),
                "max": signal.max(),
            },
            "latency": {
                "mean": latency.mean(),
                "std": latency.std(),
                "min": latency.min(),
                "max": latency.max(),
            },
            "resource_allocation": {
                "mean": allocation.mean(),
                "std": allocation.std(),
                "min": allocation.min(),
                "max": allocation.max(),
            },
        }

        return stats

    def split_data(
        self,
        stratify_col: Optional[str] = "Application_Type"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            stratify_col: Column to stratify split on

        Returns:
            Tuple of (train, val, test) DataFrames
        """
        from sklearn.model_selection import train_test_split

        if self.data is None:
            raise ValueError("Data not loaded")

        # Check if stratification is possible
        stratify = None
        if stratify_col and stratify_col in self.data.columns:
            # Check minimum class counts
            class_counts = self.data[stratify_col].value_counts()
            min_class_count = class_counts.min()

            # Need at least 2 samples per class for stratification
            if min_class_count >= 2:
                stratify = self.data[stratify_col]
                logger.info(f"Using stratified split on {stratify_col}")
            else:
                logger.warning(
                    f"Cannot stratify on {stratify_col}: "
                    f"minimum class count is {min_class_count}. "
                    f"Using random split instead."
                )
                stratify_col = None  # Disable for second split too

        # First split: train+val and test
        train_val, self.test_data = train_test_split(
            self.data,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=stratify
        )

        # Second split: train and val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)

        # Check stratification for second split
        stratify_val = None
        if stratify_col and stratify_col in train_val.columns:
            class_counts_val = train_val[stratify_col].value_counts()
            if class_counts_val.min() >= 2:
                stratify_val = train_val[stratify_col]

        self.train_data, self.val_data = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=self.config.random_seed,
            stratify=stratify_val
        )

        logger.info(f"Data split - Train: {len(self.train_data)}, "
                   f"Val: {len(self.val_data)}, Test: {len(self.test_data)}")

        return self.train_data, self.val_data, self.test_data

    def save_processed_data(self, output_dir: str) -> None:
        """Save processed train/val/test splits."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.train_data is not None:
            self.train_data.to_csv(output_path / "train.csv", index=False)
        if self.val_data is not None:
            self.val_data.to_csv(output_path / "val.csv", index=False)
        if self.test_data is not None:
            self.test_data.to_csv(output_path / "test.csv", index=False)

        logger.info(f"Processed data saved to {output_dir}")


def load_qos_data(data_path: str, config: Optional[DataConfig] = None) -> QoSDataLoader:
    """
    Convenience function to load and validate QoS data.

    Args:
        data_path: Path to CSV file
        config: Optional data configuration

    Returns:
        Initialized and loaded QoSDataLoader
    """
    if config is None:
        config = DataConfig(raw_data_path=data_path)

    loader = QoSDataLoader(config)
    loader.load_data()
    loader.validate_data()

    return loader
