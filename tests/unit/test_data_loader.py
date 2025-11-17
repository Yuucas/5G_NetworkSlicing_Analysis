"""
Unit tests for data loader module.
"""

import pandas as pd
import pytest

from src.data.data_loader import DataConfig, QoSDataLoader, load_qos_data


class TestDataConfig:
    """Test DataConfig class."""

    def test_config_initialization(self):
        """Test DataConfig initialization with defaults."""
        config = DataConfig(raw_data_path="test.csv")
        assert config.raw_data_path == "test.csv"
        assert config.test_size == 0.2
        assert config.val_size == 0.1
        assert config.random_seed == 42

    def test_config_custom_values(self):
        """Test DataConfig with custom values."""
        config = DataConfig(raw_data_path="data.csv", test_size=0.3, val_size=0.15, random_seed=123)
        assert config.test_size == 0.3
        assert config.val_size == 0.15
        assert config.random_seed == 123


class TestQoSDataLoader:
    """Test QoSDataLoader class."""

    def test_loader_initialization(self, sample_csv_file):
        """Test loader initialization."""
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        assert loader.data is None
        assert loader.train_data is None
        assert loader.val_data is None
        assert loader.test_data is None

    def test_load_data(self, sample_csv_file):
        """Test data loading."""
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        loader.load_data()

        assert loader.data is not None
        assert isinstance(loader.data, pd.DataFrame)
        assert len(loader.data) == 10
        assert all(col in loader.data.columns for col in loader.EXPECTED_COLUMNS)

    def test_load_data_missing_file(self):
        """Test loading with non-existent file."""
        config = DataConfig(raw_data_path="nonexistent.csv")
        loader = QoSDataLoader(config)

        with pytest.raises(Exception):
            loader.load_data()

    def test_validate_data(self, sample_csv_file):
        """Test data validation."""
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        loader.load_data()

        # Validation should pass (with possible warnings)
        result = loader.validate_data()
        assert isinstance(result, bool)

    def test_validate_data_before_load(self, sample_csv_file):
        """Test validation before loading data."""
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)

        with pytest.raises(ValueError, match="Data not loaded"):
            loader.validate_data()

    def test_get_statistics(self, sample_csv_file):
        """Test statistics generation."""
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        loader.load_data()

        stats = loader.get_statistics()

        assert "total_records" in stats
        assert "unique_users" in stats
        assert "application_distribution" in stats
        assert "signal_strength" in stats
        assert "latency" in stats
        assert "resource_allocation" in stats

        assert stats["total_records"] == 10
        assert stats["unique_users"] == 10

    def test_split_data(self, sample_csv_file):
        """Test data splitting."""
        config = DataConfig(
            raw_data_path=sample_csv_file, test_size=0.2, val_size=0.1, random_seed=42
        )
        loader = QoSDataLoader(config)
        loader.load_data()

        train, val, test = loader.split_data(
            stratify_col=None
        )  # No stratification for small dataset

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == 10

    def test_split_data_reproducible(self, sample_csv_file):
        """Test that data splitting is reproducible."""
        config = DataConfig(raw_data_path=sample_csv_file, random_seed=42)

        # First split
        loader1 = QoSDataLoader(config)
        loader1.load_data()
        train1, val1, test1 = loader1.split_data(stratify_col=None)

        # Second split with same seed
        loader2 = QoSDataLoader(config)
        loader2.load_data()
        train2, val2, test2 = loader2.split_data(stratify_col=None)

        # Should be identical
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_save_processed_data(self, sample_csv_file, tmp_path):
        """Test saving processed data."""
        config = DataConfig(raw_data_path=sample_csv_file)
        loader = QoSDataLoader(config)
        loader.load_data()
        loader.split_data(stratify_col=None)

        output_dir = tmp_path / "processed"
        loader.save_processed_data(str(output_dir))

        assert (output_dir / "train.csv").exists()
        assert (output_dir / "val.csv").exists()
        assert (output_dir / "test.csv").exists()


class TestLoadQoSData:
    """Test convenience function."""

    def test_load_qos_data(self, sample_csv_file):
        """Test load_qos_data convenience function."""
        loader = load_qos_data(sample_csv_file)

        assert loader is not None
        assert isinstance(loader, QoSDataLoader)
        assert loader.data is not None
        assert len(loader.data) == 10

    def test_load_qos_data_with_config(self, sample_csv_file):
        """Test load_qos_data with custom config."""
        config = DataConfig(raw_data_path=sample_csv_file, random_seed=123)
        loader = load_qos_data(sample_csv_file, config)

        assert loader.config.random_seed == 123
