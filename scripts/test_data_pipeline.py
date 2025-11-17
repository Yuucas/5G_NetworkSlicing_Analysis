"""
Test script to verify data pipeline works correctly.
This tests data loading and preprocessing without ML dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import QoSDataLoader, DataConfig
from src.data.preprocessing import QoSPreprocessor

def main():
    print("=" * 60)
    print("Testing 5G Network Slicing Data Pipeline")
    print("=" * 60)

    # Configuration
    data_path = "data/raw/Quality of Service 5G.csv"
    print(f"\n1. Loading data from: {data_path}")

    # Load data
    data_config = DataConfig(raw_data_path=data_path)
    loader = QoSDataLoader(data_config)
    loader.load_data()

    print(f"   [OK] Loaded {len(loader.data)} records")

    # Validate data
    print("\n2. Validating data...")
    is_valid = loader.validate_data()
    print(f"   [OK] Validation {'passed' if is_valid else 'failed with warnings'}")

    # Get statistics
    print("\n3. Dataset statistics:")
    stats = loader.get_statistics()
    print(f"   - Total records: {stats['total_records']}")
    print(f"   - Unique users: {stats['unique_users']}")
    print(f"   - Application types: {len(stats['application_distribution'])}")
    print(f"   - Signal strength: {stats['signal_strength']['mean']:.2f} dBm (avg)")
    print(f"   - Latency: {stats['latency']['mean']:.2f} ms (avg)")
    print(f"   - Resource allocation: {stats['resource_allocation']['mean']:.2f}% (avg)")

    print("\n4. Application distribution:")
    for app_type, count in sorted(stats['application_distribution'].items(),
                                   key=lambda x: x[1], reverse=True)[:5]:
        print(f"   - {app_type}: {count} requests")

    # Split data
    print("\n5. Splitting data (train/val/test)...")
    train_data, val_data, test_data = loader.split_data()
    print(f"   [OK] Train: {len(train_data)} samples")
    print(f"   [OK] Val: {len(val_data)} samples")
    print(f"   [OK] Test: {len(test_data)} samples")

    # Preprocess
    print("\n6. Preprocessing data...")
    preprocessor = QoSPreprocessor(
        scaler_type="standard",
        encode_categorical=True,
        extract_time_features=True
    )

    train_processed = preprocessor.fit_transform(train_data)
    print(f"   [OK] Processed training data: {train_processed.shape}")
    print(f"   [OK] Features created: {len(train_processed.columns)}")

    val_processed = preprocessor.transform(val_data)
    print(f"   [OK] Processed validation data: {val_processed.shape}")

    # Show processed columns
    print("\n7. Processed features:")
    for i, col in enumerate(train_processed.columns[:15], 1):
        print(f"   {i}. {col}")
    if len(train_processed.columns) > 15:
        print(f"   ... and {len(train_processed.columns) - 15} more")

    # Save processed data
    print("\n8. Saving processed data...")
    output_dir = "data/processed"
    loader.save_processed_data(output_dir)
    print(f"   [OK] Saved to: {output_dir}/")

    print("\n" + "=" * 60)
    print("[OK] Data pipeline test completed successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
