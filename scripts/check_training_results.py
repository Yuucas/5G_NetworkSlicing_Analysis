"""
Check training results and checkpoint files.
Works without ML dependencies.
"""

import sys
from pathlib import Path
import os

def check_results():
    """Check what files were created during training."""

    print("=" * 60)
    print("Checking Training Results")
    print("=" * 60)

    base_path = Path(".")

    # Check checkpoints
    print("\n1. Model Checkpoints:")
    checkpoint_dir = base_path / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if checkpoint_files:
            print(f"   [OK] Found {len(checkpoint_files)} checkpoint(s):")
            for f in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True):
                size_mb = f.stat().st_size / (1024 * 1024)
                mtime = f.stat().st_mtime
                import time
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                print(f"        - {f.name} ({size_mb:.2f} MB) - {time_str}")
        else:
            print("   [INFO] No checkpoints found (.pt files)")
    else:
        print("   [INFO] Checkpoints directory doesn't exist")
        print("   [ACTION] Create it: mkdir checkpoints")

    # Check logs
    print("\n2. Training Logs:")
    logs_dir = base_path / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            print(f"   [OK] Found {len(log_files)} log file(s):")
            for f in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True):
                size_kb = f.stat().st_size / 1024
                print(f"        - {f.name} ({size_kb:.1f} KB)")
        else:
            print("   [INFO] No log files found")
    else:
        print("   [INFO] Logs directory doesn't exist")

    # Check processed data
    print("\n3. Processed Data:")
    data_dir = base_path / "data" / "processed"
    if data_dir.exists():
        data_files = list(data_dir.glob("*.csv"))
        if data_files:
            print(f"   [OK] Found {len(data_files)} processed file(s):")
            for f in sorted(data_files):
                size_kb = f.stat().st_size / 1024
                print(f"        - {f.name} ({size_kb:.1f} KB)")

                # Count rows
                try:
                    with open(f, 'r') as file:
                        row_count = sum(1 for line in file) - 1  # Subtract header
                    print(f"          Rows: {row_count}")
                except:
                    pass
        else:
            print("   [INFO] No processed data files found")
    else:
        print("   [INFO] Processed data directory doesn't exist")

    # Check MLflow runs
    print("\n4. MLflow Experiment Tracking:")
    mlruns_dir = base_path / "mlruns"
    if mlruns_dir.exists():
        experiments = [d for d in mlruns_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if experiments:
            print(f"   [OK] Found {len(experiments)} experiment(s)")
            for exp in experiments:
                runs = [d for d in exp.iterdir() if d.is_dir() and not d.name.startswith('.')]
                print(f"        - Experiment {exp.name}: {len(runs)} run(s)")
        else:
            print("   [INFO] No experiments found")
    else:
        print("   [INFO] MLflow directory doesn't exist")

    # Summary and recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    has_checkpoints = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt"))

    if has_checkpoints:
        print("\n[OK] Training artifacts found!")
        print("\nNext steps:")
        print("  1. Evaluate the model:")
        print("     python scripts/evaluate_model.py --episodes 10")
        print("\n  2. Visualize results:")
        print("     python scripts/visualize_results.py")
        print("\n  3. Run demo:")
        print("     python notebooks/01_model_demo.py")
    else:
        print("\n[INFO] No training artifacts found yet.")
        print("\nTo train a model:")
        print("  python scripts/train_rl_agent.py --episodes 100")
        print("\nOr test the data pipeline:")
        print("  python scripts/test_data_pipeline.py")

    # File structure summary
    print("\n" + "=" * 60)
    print("PROJECT FILE STRUCTURE")
    print("=" * 60)

    structure = {
        "checkpoints/": "Trained model weights (.pt files)",
        "data/raw/": "Original dataset",
        "data/processed/": "Preprocessed train/val/test splits",
        "logs/": "Training and application logs",
        "mlruns/": "MLflow experiment tracking",
        "results/": "Evaluation results and plots",
    }

    for path, description in structure.items():
        full_path = base_path / path
        exists = "[OK]" if full_path.exists() else "[  ]"
        print(f"{exists} {path:<20} - {description}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        check_results()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
