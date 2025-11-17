# Test Suite Documentation

## Overview

Comprehensive test suite for the 5G Network Slicing Analysis project covering:
- Unit tests for individual components
- Integration tests for complete workflows
- API endpoint tests
- Model tests

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_data_loader.py      # Data loading tests
│   ├── test_preprocessing.py    # Preprocessing tests
│   ├── test_models.py           # Model tests (RL, DQN)
│   └── test_api.py              # API endpoint tests
└── integration/             # Integration tests
    └── test_pipeline.py         # End-to-end pipeline tests
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_data_loader.py

# Specific test class
pytest tests/unit/test_data_loader.py::TestQoSDataLoader

# Specific test function
pytest tests/unit/test_data_loader.py::TestQoSDataLoader::test_load_data
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# View coverage in browser
# Open: htmlcov/index.html
```

### Run with Verbose Output

```bash
pytest -v tests/
```

### Run Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## Test Dependencies

### Minimal (Core Tests)
```bash
pip install pytest pandas numpy scikit-learn
```

### Full (All Tests)
```bash
pip install pytest pytest-cov torch gymnasium fastapi httpx
```

## Test Coverage

### Unit Tests

#### Data Processing (test_data_loader.py, test_preprocessing.py)
- ✅ Data loading from CSV
- ✅ Data validation
- ✅ Train/val/test splitting
- ✅ Statistics generation
- ✅ Signal strength parsing
- ✅ Latency parsing
- ✅ Bandwidth conversion
- ✅ Resource allocation parsing
- ✅ Time feature extraction
- ✅ Categorical encoding
- ✅ Sequence creation

#### Models (test_models.py)
- ✅ Environment initialization
- ✅ Environment reset/step
- ✅ Reward calculation
- ✅ DQN network architecture
- ✅ Replay buffer
- ✅ Agent action selection
- ✅ Agent training
- ✅ Save/load checkpoints

#### API (test_api.py)
- ✅ Health checks
- ✅ Prediction endpoints
- ✅ Allocation endpoints
- ✅ Monitoring endpoints
- ✅ Error handling

### Integration Tests (test_pipeline.py)
- ✅ Complete data pipeline
- ✅ Data to model conversion
- ✅ Environment with real data
- ✅ Agent-environment interaction
- ✅ Mini training loop
- ✅ Save/load workflows

## Fixtures

Common fixtures available in `conftest.py`:

- **`sample_qos_data`**: Sample DataFrame with QoS data
- **`sample_csv_file`**: Temporary CSV file
- **`sample_numeric_array`**: Random numeric array for models
- **`processed_data`**: Preprocessed DataFrame
- **`test_config`**: Test configuration dict

## Writing New Tests

### Example Unit Test

```python
# tests/unit/test_my_module.py
import pytest
from src.my_module import MyClass

class TestMyClass:
    """Test MyClass functionality."""

    def test_initialization(self):
        """Test object initialization."""
        obj = MyClass(param=10)
        assert obj.param == 10

    def test_method(self, sample_qos_data):
        """Test a method using fixture."""
        obj = MyClass()
        result = obj.process(sample_qos_data)
        assert result is not None
```

### Example Integration Test

```python
# tests/integration/test_my_workflow.py
import pytest

@pytest.mark.integration
class TestMyWorkflow:
    """Test complete workflow."""

    def test_end_to_end(self):
        """Test complete process."""
        # Setup
        # Process
        # Assert
        pass
```

## CI/CD Integration

Tests are automatically run in CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: pytest tests/ --cov=src
```

## Test Metrics

Current test coverage:
- **Data Processing**: ~95%
- **Models**: ~85% (requires torch/gymnasium)
- **API**: ~90% (requires fastapi)
- **Integration**: ~80%

## Common Issues

### Issue: ModuleNotFoundError
**Solution**: Install test dependencies
```bash
pip install pytest pandas numpy scikit-learn
```

### Issue: Tests skipped
**Reason**: Missing optional dependencies (torch, gymnasium, fastapi)
**Solution**: Install for full test coverage
```bash
pip install torch gymnasium fastapi httpx
```

### Issue: Fixture not found
**Solution**: Check `conftest.py` is in tests/ directory

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Use Fixtures**: Reuse common test data
3. **Clear Names**: Test names should describe what they test
4. **One Assert**: Prefer one assertion per test when possible
5. **Mock External**: Mock external dependencies
6. **Fast Tests**: Keep unit tests fast (<1 second)

## Debugging Tests

```bash
# Run with debug output
pytest -vv --tb=long

# Run specific test with print statements
pytest -s tests/unit/test_data_loader.py::test_specific

# Drop into debugger on failure
pytest --pdb

# Run last failed tests
pytest --lf
```

## Test Reports

Generate HTML report:
```bash
pytest --html=report.html --self-contained-html
```

Generate JUnit XML (for CI):
```bash
pytest --junitxml=junit.xml
```

## Contributing Tests

When adding new features:
1. Write tests first (TDD)
2. Ensure >80% coverage
3. Include both unit and integration tests
4. Update this README if needed

## Running Specific Test Suites

```bash
# Data processing tests only
pytest tests/unit/test_data_loader.py tests/unit/test_preprocessing.py

# Model tests only
pytest tests/unit/test_models.py

# API tests only
pytest tests/unit/test_api.py

# All integration tests
pytest tests/integration/
```

## Performance Testing

For performance-critical tests:
```bash
pytest --durations=10  # Show 10 slowest tests
```

## Contact

For test-related questions, see main README or open an issue.
