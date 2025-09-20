# 🧪 Violet Model Merge - Test Suite

Comprehensive test coverage for the Violet Model Merge project, ensuring reliability and maintainability as we move fast and break things! 😅

## 📋 Test Structure

```
tests/
├── 📄 conftest.py              # Test configuration and fixtures
├── 🧪 test_smoke.py           # Basic smoke tests
├── 🔧 test_utils.py           # Unit tests for utility functions
├── 🎯 test_merge_model.py     # Core merge engine tests
├── 💅 test_vae.py             # VAE operations tests
├── 💻 test_cli.py             # CLI interface tests
└── 📊 htmlcov/                # Coverage reports (generated)
```

## 🚀 Quick Start

### Install Test Dependencies

```bash
# Install test dependencies
pip install -e ".[test]"

# Or install all development dependencies
pip install -e ".[dev,test,lint]"
```

### Run Tests

```bash
# Run all tests with coverage
python run_tests.py

# Run only fast tests
python run_tests.py --fast

# Run with pytest directly
pytest tests/ --cov=lib --cov-report=html -v
```

## 🎯 Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **test_utils.py**: Core utility functions
- **test_merge_model.py**: Merge algorithms and model operations
- **test_vae.py**: VAE merging and baking functionality

### Integration Tests (`@pytest.mark.integration`)
- **test_cli.py**: Command-line interface workflows
- Complete end-to-end merge scenarios
- File I/O and error handling

### Performance Tests (`@pytest.mark.slow`)
- Memory usage validation
- Large model handling
- Performance benchmarks

## 🔧 Test Configuration

### Markers
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Slower integration tests  
- `@pytest.mark.slow` - Performance and stress tests
- `@pytest.mark.gpu` - GPU-dependent tests (skipped in CI)

### Fixtures
- `temp_dir` - Temporary directory for test files
- `mock_safetensors_model` - Mock model structure
- `mock_model_files` - Complete mock model files
- `sample_merge_config` - Default merge configuration

## 📊 Coverage Goals

Our test suite aims for:
- **>80% overall coverage** - Comprehensive code coverage
- **>90% core functionality** - Critical merge algorithms
- **>70% edge cases** - Error handling and validation

Current coverage targets by module:
- `lib/utils.py` - 85%+
- `lib/merge_model.py` - 90%+
- `lib/merge_vae.py` - 80%+

## 🏃‍♂️ Running Tests Locally

### Quick Test Run
```bash
# Fast development feedback
python run_tests.py --fast --no-coverage
```

### Full Test Suite
```bash
# Complete test run with coverage
python run_tests.py --coverage --html-report
```

### Specific Test Categories
```bash
# Only unit tests
python run_tests.py --unit-only

# Only integration tests  
python run_tests.py --integration-only

# Code quality checks
python run_tests.py --lint
```

### Parallel Testing
```bash
# Run tests in parallel (faster)
python run_tests.py --parallel 4
```

## 🔍 Code Quality

### Formatting and Linting
```bash
# Check code formatting
black --check lib/ tests/
isort --check-only lib/ tests/

# Lint code
flake8 lib/ tests/

# Type checking
mypy lib/ --ignore-missing-imports
```

### Security Analysis
```bash
# Security vulnerability scanning
bandit -r lib/
safety check
```

## 🧪 Mock Data and Fixtures

The test suite uses sophisticated mocking to avoid dependencies on large model files:

### Mock Models
- **Safetensors structure**: Realistic tensor shapes and keys
- **Multiple architectures**: SD1.5, SDXL, Flux support
- **VAE components**: Separate encoder/decoder structures

### Test Data
- **Deterministic**: Same results across runs
- **Lightweight**: Fast test execution
- **Comprehensive**: Cover edge cases and error conditions

## 🎯 Writing New Tests

### Test Organization
```python
@pytest.mark.unit
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_operation(self):
        """Test basic feature operation."""
        # Arrange
        input_data = create_test_data()
        
        # Act  
        result = new_feature(input_data)
        
        # Assert
        assert result.is_valid()
```

### Best Practices
1. **Use descriptive names**: `test_weighted_sum_with_different_dtypes`
2. **Test edge cases**: Empty inputs, invalid parameters, etc.
3. **Mock external dependencies**: File I/O, CUDA, etc.
4. **Use fixtures**: Reusable test data and setup
5. **Add markers**: Categorize tests appropriately

### Mock Examples
```python
def test_model_loading(mock_model_files):
    """Test model loading with mock files."""
    model = load_model(mock_model_files['model_a'])
    assert model is not None
    assert len(model) > 0

@patch('torch.cuda.is_available', return_value=False)
def test_cpu_fallback(mock_cuda):
    """Test CPU fallback when CUDA unavailable."""
    device = detect_device('auto')
    assert device == 'cpu'
```

## 🚀 Continuous Integration

### GitHub Actions
Our CI pipeline automatically:
- ✅ Tests across Python 3.8-3.12
- ✅ Tests on Windows, macOS, Linux  
- ✅ Generates coverage reports
- ✅ Runs security scans
- ✅ Validates code quality

### Coverage Reports
- **Codecov integration**: Automatic coverage tracking
- **HTML reports**: Detailed line-by-line coverage
- **Trend analysis**: Coverage changes over time

## 🐛 Debugging Tests

### Verbose Output
```bash
# See detailed test output
pytest tests/ -v -s --tb=long

# Show print statements
pytest tests/ -s
```

### Debug Specific Tests
```bash
# Run single test
pytest tests/test_utils.py::test_wgt_basic_operations -v

# Run tests matching pattern
pytest tests/ -k "test_merge" -v
```

### Coverage Analysis
```bash
# Generate detailed coverage report
pytest tests/ --cov=lib --cov-report=html
# Open htmlcov/index.html in browser
```

## 💜 Contributing

When adding new features:
1. **Write tests first** (TDD approach)
2. **Ensure good coverage** (>80% for new code)
3. **Add integration tests** for user-facing features
4. **Update this README** if adding new test categories

Remember: Good tests are the foundation for moving fast and breaking things safely! 🎯✨

---

**Built with 💜 for the AI art community** — *Test with confidence, merge with precision*