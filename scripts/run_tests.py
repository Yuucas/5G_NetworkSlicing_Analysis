"""
Test runner script with different test modes.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print("=" * 60)
    print(f"{description}")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n[OK] {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed!")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n[ERROR] pytest not found. Install it with: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests with different modes")
    parser.add_argument(
        "--mode",
        choices=["all", "unit", "integration", "coverage", "fast", "api", "models", "data"],
        default="all",
        help="Test mode to run"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific test file to run"
    )

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["pytest"]

    if args.verbose:
        base_cmd.append("-v")

    # Mode-specific commands
    if args.mode == "all":
        cmd = base_cmd + ["tests/"]
        run_command(cmd, "Running All Tests")

    elif args.mode == "unit":
        cmd = base_cmd + ["tests/unit/"]
        run_command(cmd, "Running Unit Tests")

    elif args.mode == "integration":
        cmd = base_cmd + ["tests/integration/"]
        run_command(cmd, "Running Integration Tests")

    elif args.mode == "coverage":
        cmd = base_cmd + [
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html"
        ]
        success = run_command(cmd, "Running Tests with Coverage")
        if success:
            print("\n" + "=" * 60)
            print("Coverage report generated!")
            print("View HTML report: htmlcov/index.html")
            print("=" * 60)

    elif args.mode == "fast":
        cmd = base_cmd + ["tests/unit/", "-m", "not slow"]
        run_command(cmd, "Running Fast Tests Only")

    elif args.mode == "api":
        cmd = base_cmd + ["tests/unit/test_api.py"]
        run_command(cmd, "Running API Tests")

    elif args.mode == "models":
        cmd = base_cmd + ["tests/unit/test_models.py"]
        run_command(cmd, "Running Model Tests")

    elif args.mode == "data":
        cmd = base_cmd + ["tests/unit/test_data_loader.py", "tests/unit/test_preprocessing.py"]
        run_command(cmd, "Running Data Processing Tests")

    elif args.file:
        cmd = base_cmd + [args.file]
        run_command(cmd, f"Running Tests in {args.file}")


if __name__ == "__main__":
    main()
