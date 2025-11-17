"""
Data preprocessing and transformation for 5G QoS data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class QoSPreprocessor:
    """
    Comprehensive preprocessor for 5G QoS data.

    Handles:
    - Unit conversion (dBm, Kbps/Mbps, ms)
    - Feature scaling and normalization
    - Categorical encoding
    - Time-based feature extraction
    - Missing value imputation
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        encode_categorical: bool = True,
        extract_time_features: bool = True,
    ):
        self.scaler_type = scaler_type
        self.encode_categorical = encode_categorical
        self.extract_time_features = extract_time_features

        self.scaler = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []

    def fit(self, data: pd.DataFrame) -> "QoSPreprocessor":
        """Fit preprocessor on training data."""
        self.processed_data = data.copy()

        # Parse and convert units
        self._parse_signal_strength()
        self._parse_latency()
        self._parse_bandwidth()
        self._parse_resource_allocation()

        # Extract time features
        if self.extract_time_features:
            self._extract_time_features()

        # Encode categorical variables
        if self.encode_categorical:
            self._fit_label_encoders()

        # Fit scaler on numeric features
        self._fit_scaler()

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        transformed = data.copy()

        # Apply same transformations
        self._parse_signal_strength(transformed)
        self._parse_latency(transformed)
        self._parse_bandwidth(transformed)
        self._parse_resource_allocation(transformed)

        if self.extract_time_features:
            self._extract_time_features(transformed)

        if self.encode_categorical:
            self._transform_categorical(transformed)

        self._scale_features(transformed)

        return transformed

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def _parse_signal_strength(self, data: Optional[pd.DataFrame] = None) -> None:
        """Parse signal strength from string format (-XX dBm) to float."""
        df = data if data is not None else self.processed_data

        if "Signal_Strength" in df.columns:
            # Extract numeric value from "-XX dBm" format
            df["Signal_Strength_dBm"] = (
                df["Signal_Strength"].str.extract(r"(-?\d+)")[0].astype(float)
            )

            # Normalize to [0, 1] range (typical range: -120 to -30 dBm)
            # Better signal = higher value
            df["Signal_Quality"] = (df["Signal_Strength_dBm"] + 120) / 90

            df.drop("Signal_Strength", axis=1, inplace=True)

    def _parse_latency(self, data: Optional[pd.DataFrame] = None) -> None:
        """Parse latency from string format (XX ms) to float."""
        df = data if data is not None else self.processed_data

        if "Latency" in df.columns:
            df["Latency_ms"] = df["Latency"].str.extract(r"(\d+)")[0].astype(float)
            df.drop("Latency", axis=1, inplace=True)

    def _parse_bandwidth(self, data: Optional[pd.DataFrame] = None) -> None:
        """Parse and convert bandwidth to Mbps."""
        df = data if data is not None else self.processed_data

        for col in ["Required_Bandwidth", "Allocated_Bandwidth"]:
            if col in df.columns:
                # Extract number and unit
                bandwidth_data = df[col].str.extract(r"([\d.]+)\s*(Kbps|Mbps)")

                value = bandwidth_data[0].astype(float)
                unit = bandwidth_data[1]

                # Convert to Mbps
                mbps_value = value.copy()
                mbps_value[unit == "Kbps"] = value[unit == "Kbps"] / 1000

                df[f"{col}_Mbps"] = mbps_value
                df.drop(col, axis=1, inplace=True)

        # Calculate bandwidth utilization ratio
        if "Required_Bandwidth_Mbps" in df.columns and "Allocated_Bandwidth_Mbps" in df.columns:
            df["Bandwidth_Utilization"] = (
                df["Required_Bandwidth_Mbps"] / df["Allocated_Bandwidth_Mbps"].replace(0, np.nan)
            ).fillna(1.0)

            # Overallocation indicator
            df["Is_Overallocated"] = (
                df["Allocated_Bandwidth_Mbps"] > df["Required_Bandwidth_Mbps"]
            ).astype(int)

    def _parse_resource_allocation(self, data: Optional[pd.DataFrame] = None) -> None:
        """Parse resource allocation from percentage string to float."""
        df = data if data is not None else self.processed_data

        if "Resource_Allocation" in df.columns:
            df["Resource_Allocation_Pct"] = df["Resource_Allocation"].str.rstrip("%").astype(float)
            df.drop("Resource_Allocation", axis=1, inplace=True)

    def _extract_time_features(self, data: Optional[pd.DataFrame] = None) -> None:
        """Extract time-based features from timestamp."""
        df = data if data is not None else self.processed_data

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            # Cyclical time features
            df["Hour"] = df["Timestamp"].dt.hour
            df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
            df["DayOfMonth"] = df["Timestamp"].dt.day

            # Cyclical encoding for hour and day of week
            df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
            df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
            df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
            df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

            # Peak hour indicator (8-10am, 6-9pm typical peak hours)
            df["Is_Peak_Hour"] = df["Hour"].isin([8, 9, 18, 19, 20]).astype(int)

            # Keep timestamp for sorting but drop from features
            # df.drop("Timestamp", axis=1, inplace=True)

    def _fit_label_encoders(self) -> None:
        """Fit label encoders for categorical variables."""
        categorical_cols = ["Application_Type", "User_ID"]

        for col in categorical_cols:
            if col in self.processed_data.columns:
                le = LabelEncoder()
                le.fit(self.processed_data[col])
                self.label_encoders[col] = le

    def _transform_categorical(self, data: pd.DataFrame) -> None:
        """Transform categorical variables using fitted encoders."""
        for col, encoder in self.label_encoders.items():
            if col in data.columns:
                # Handle unseen categories
                data[f"{col}_Encoded"] = data[col].map(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
                data.drop(col, axis=1, inplace=True)

    def _fit_scaler(self) -> None:
        """Fit scaler on numeric features."""
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target variable
        if "Resource_Allocation_Pct" in numeric_cols:
            numeric_cols.remove("Resource_Allocation_Pct")

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        self.scaler.fit(self.processed_data[numeric_cols])
        self.feature_names = numeric_cols

    def _scale_features(self, data: pd.DataFrame) -> None:
        """Scale numeric features using fitted scaler."""
        if self.scaler and self.feature_names:
            data[self.feature_names] = self.scaler.transform(data[self.feature_names])

    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names after preprocessing."""
        return self.feature_names


class TimeSeriesPreprocessor:
    """
    Preprocessor for time-series specific transformations.

    Creates sequences and handles temporal dependencies.
    """

    def __init__(self, sequence_length: int = 10, prediction_horizon: int = 1, stride: int = 1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride

    def create_sequences(
        self,
        data: pd.DataFrame,
        target_col: str = "Resource_Allocation_Pct",
        group_by: Optional[str] = "User_ID_Encoded",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series modeling.

        Args:
            data: Preprocessed dataframe
            target_col: Target column name
            group_by: Column to group sequences by (e.g., User_ID)

        Returns:
            Tuple of (X_sequences, y_targets)
        """
        X_sequences = []
        y_targets = []

        if group_by:
            # Create sequences per user/group
            for group_id in data[group_by].unique():
                group_data = data[data[group_by] == group_id].sort_values("Timestamp")
                X_seq, y_seq = self._create_group_sequences(group_data, target_col)
                X_sequences.extend(X_seq)
                y_targets.extend(y_seq)
        else:
            # Create sequences for entire dataset
            X_sequences, y_targets = self._create_group_sequences(data, target_col)

        return np.array(X_sequences), np.array(y_targets)

    def _create_group_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[List, List]:
        """Create sequences for a single group."""
        feature_cols = [
            col for col in data.columns if col not in [target_col, "Timestamp", "User_ID"]
        ]

        X = data[feature_cols].values
        y = data[target_col].values

        X_seq = []
        y_seq = []

        for i in range(0, len(X) - self.sequence_length - self.prediction_horizon + 1, self.stride):
            X_seq.append(X[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length + self.prediction_horizon - 1])

        return X_seq, y_seq
