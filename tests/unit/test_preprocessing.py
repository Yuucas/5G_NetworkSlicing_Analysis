"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import QoSPreprocessor, TimeSeriesPreprocessor


class TestQoSPreprocessor:
    """Test QoSPreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = QoSPreprocessor()
        assert preprocessor.scaler_type == "standard"
        assert preprocessor.encode_categorical is True
        assert preprocessor.extract_time_features is True

    def test_preprocessor_custom_config(self):
        """Test preprocessor with custom configuration."""
        preprocessor = QoSPreprocessor(
            scaler_type="robust",
            encode_categorical=False,
            extract_time_features=False
        )
        assert preprocessor.scaler_type == "robust"
        assert preprocessor.encode_categorical is False
        assert preprocessor.extract_time_features is False

    def test_fit_transform(self, sample_qos_data):
        """Test fit_transform method."""
        preprocessor = QoSPreprocessor()
        result = preprocessor.fit_transform(sample_qos_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_qos_data)
        assert preprocessor.scaler is not None

    def test_signal_strength_parsing(self, sample_qos_data):
        """Test signal strength parsing."""
        preprocessor = QoSPreprocessor()
        result = preprocessor.fit_transform(sample_qos_data)

        assert 'Signal_Strength_dBm' in result.columns
        assert 'Signal_Quality' in result.columns
        assert 'Signal_Strength' not in result.columns

        # Check values are numeric
        assert result['Signal_Strength_dBm'].dtype in [np.float64, np.float32, np.int64]
        assert result['Signal_Quality'].dtype in [np.float64, np.float32]

    def test_latency_parsing(self, sample_qos_data):
        """Test latency parsing."""
        preprocessor = QoSPreprocessor()
        result = preprocessor.fit_transform(sample_qos_data)

        assert 'Latency_ms' in result.columns
        assert 'Latency' not in result.columns
        assert result['Latency_ms'].dtype in [np.float64, np.float32, np.int64]

    def test_bandwidth_parsing(self, sample_qos_data):
        """Test bandwidth parsing and conversion."""
        preprocessor = QoSPreprocessor()
        result = preprocessor.fit_transform(sample_qos_data)

        assert 'Required_Bandwidth_Mbps' in result.columns
        assert 'Allocated_Bandwidth_Mbps' in result.columns
        assert 'Bandwidth_Utilization' in result.columns
        assert 'Is_Overallocated' in result.columns

        # Check utilization is numeric (after scaling, can be negative)
        assert result['Bandwidth_Utilization'].dtype in [np.float64, np.float32]
        assert not result['Bandwidth_Utilization'].isna().any()

    def test_resource_allocation_parsing(self, sample_qos_data):
        """Test resource allocation parsing."""
        preprocessor = QoSPreprocessor()
        result = preprocessor.fit_transform(sample_qos_data)

        assert 'Resource_Allocation_Pct' in result.columns
        assert 'Resource_Allocation' not in result.columns

        # Check values are in valid range
        assert (result['Resource_Allocation_Pct'] >= 0).all()
        assert (result['Resource_Allocation_Pct'] <= 100).all()

    def test_time_feature_extraction(self, sample_qos_data):
        """Test time feature extraction."""
        preprocessor = QoSPreprocessor(extract_time_features=True)
        result = preprocessor.fit_transform(sample_qos_data)

        expected_time_features = [
            'Hour', 'DayOfWeek', 'DayOfMonth',
            'Hour_Sin', 'Hour_Cos',
            'DayOfWeek_Sin', 'DayOfWeek_Cos',
            'Is_Peak_Hour'
        ]

        for feature in expected_time_features:
            assert feature in result.columns

        # Check cyclical encoding bounds
        assert (result['Hour_Sin'] >= -1).all() and (result['Hour_Sin'] <= 1).all()
        assert (result['Hour_Cos'] >= -1).all() and (result['Hour_Cos'] <= 1).all()

    def test_categorical_encoding(self, sample_qos_data):
        """Test categorical encoding."""
        preprocessor = QoSPreprocessor(encode_categorical=True)
        result = preprocessor.fit_transform(sample_qos_data)

        assert 'Application_Type_Encoded' in result.columns
        assert 'User_ID_Encoded' in result.columns
        assert 'Application_Type' not in result.columns
        assert 'User_ID' not in result.columns

    def test_transform_consistency(self, sample_qos_data):
        """Test that transform uses fitted parameters."""
        preprocessor = QoSPreprocessor()

        # Fit on all data
        preprocessor.fit(sample_qos_data)

        # Transform first half
        result1 = preprocessor.transform(sample_qos_data.iloc[:5])
        assert len(result1) == 5

        # Transform should work on new data
        result2 = preprocessor.transform(sample_qos_data.iloc[5:])
        assert len(result2) == 5

    def test_get_feature_names(self, sample_qos_data):
        """Test getting feature names."""
        preprocessor = QoSPreprocessor()
        preprocessor.fit_transform(sample_qos_data)

        feature_names = preprocessor.get_feature_importance_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0


class TestTimeSeriesPreprocessor:
    """Test TimeSeriesPreprocessor class."""

    def test_time_series_init(self):
        """Test TimeSeriesPreprocessor initialization."""
        preprocessor = TimeSeriesPreprocessor(
            sequence_length=10,
            prediction_horizon=1,
            stride=1
        )
        assert preprocessor.sequence_length == 10
        assert preprocessor.prediction_horizon == 1
        assert preprocessor.stride == 1

    def test_create_sequences(self, processed_data):
        """Test sequence creation."""
        preprocessor = TimeSeriesPreprocessor(sequence_length=5, prediction_horizon=1)

        X, y = preprocessor.create_sequences(
            processed_data,
            target_col='Resource_Allocation_Pct',
            group_by=None
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[1] == 5  # sequence length
        assert len(X) == len(y)

    def test_sequences_shape(self, processed_data):
        """Test that sequences have correct shape."""
        seq_len = 3
        preprocessor = TimeSeriesPreprocessor(sequence_length=seq_len)

        X, y = preprocessor.create_sequences(
            processed_data,
            target_col='Resource_Allocation_Pct',
            group_by=None
        )

        # X should be (num_sequences, seq_len, num_features)
        assert X.ndim == 3
        assert X.shape[1] == seq_len

        # y should be (num_sequences,)
        assert y.ndim == 1

    def test_different_strides(self, processed_data):
        """Test sequence creation with different strides."""
        preprocessor1 = TimeSeriesPreprocessor(sequence_length=5, stride=1)
        preprocessor2 = TimeSeriesPreprocessor(sequence_length=5, stride=2)

        X1, y1 = preprocessor1.create_sequences(processed_data, group_by=None)
        X2, y2 = preprocessor2.create_sequences(processed_data, group_by=None)

        # Stride 2 should produce fewer sequences
        assert len(X2) < len(X1)
