# nf4ad Test Suite

This directory contains the test suite for the nf4ad package.

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_vaeflow.py
```

### Run specific test class or function
```bash
pytest tests/test_vaeflow.py::TestVAEFlow
pytest tests/test_vaeflow.py::TestVAEFlow::test_forward_pass
```

### Run with coverage
```bash
pytest --cov=nf4ad --cov-report=html
```

### Skip slow tests
```bash
pytest -m "not slow"
```

### Run only GPU tests
```bash
pytest -m gpu
```

## Test Structure

- `conftest.py`: Shared fixtures and configuration
- `test_vaeflow.py`: Tests for VAEFlow core components
- `test_adbench_wrapper.py`: Tests for ADBench wrapper
- `test_flows.py`: Tests for flow components

## Fixtures

Common fixtures available in all tests:

- `device`: Auto-detected device (CUDA/MPS/CPU)
- `synthetic_image_data`: Synthetic image dataset
- `synthetic_tabular_data`: Synthetic tabular dataset
- `simple_flow_prior`: Pre-configured flow for testing
- `small_dataset`: Small torch dataset

## Markers

- `@pytest.mark.slow`: Slow tests (skip with `-m "not slow"`)
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.integration`: Integration tests

## Adding New Tests

1. Create new test file following naming convention: `test_*.py`
2. Create test classes with prefix `Test*`
3. Create test functions with prefix `test_*`
4. Use fixtures from `conftest.py`
5. Add appropriate markers for slow/GPU tests

Example:
```python
import pytest

class TestMyComponent:
    def test_initialization(self):
        # Test code
        pass
    
    @pytest.mark.slow
    def test_training(self):
        # Slow test code
        pass
```
