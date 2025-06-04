# GLLM Balancer Tests

This directory contains comprehensive unit and integration tests for the GLLM balancer service.

## Test Structure

### Test Files

- **`test_balancer.py`** - Complete unittest-based test suite
- **`test_balancer_pytest.py`** - Modern pytest-based test suite (recommended)
- **`test_integration.py`** - Integration tests with multi-worker scenarios
- **`conftest.py`** - Pytest configuration and shared fixtures

### Test Categories

1. **Unit Tests** - Test individual components and functions
2. **Integration Tests** - Test complete workflows and multi-component interactions
3. **Load Balancing Tests** - Verify request distribution algorithms
4. **Error Handling Tests** - Test various failure scenarios
5. **Concurrency Tests** - Test thread safety and concurrent request handling

## Running the Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements.txt
```

### Running All Tests

Using pytest (recommended):
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=gllm.balancer tests/
```

Using unittest:
```bash
# Run unittest version
python -m unittest tests.test_balancer

# Run with verbose output
python -m unittest -v tests.test_balancer
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest -m unit tests/

# Run only integration tests
pytest -m integration tests/

# Run only fast tests (exclude slow tests)
pytest -m "not slow" tests/

# Run specific test file
pytest tests/test_balancer_pytest.py

# Run specific test class
pytest tests/test_balancer_pytest.py::TestWorkerManagement

# Run specific test method
pytest tests/test_balancer_pytest.py::TestWorkerManagement::test_add_worker_success
```

## Test Coverage

The test suite covers the following aspects of the balancer service:

### Core Functionality
- ✅ Worker registration and management
- ✅ Load balancing algorithm (least busy worker selection)
- ✅ Request routing to workers
- ✅ Health check endpoint
- ✅ Model loading across workers
- ✅ GPU resource management

### API Endpoints
- ✅ `/health` - Health check
- ✅ `/add_worker` - Worker registration
- ✅ `/chat/completions` - Chat completions
- ✅ `/completions` - Text completions
- ✅ `/load_model` - Model loading
- ✅ `/release_gpus` - GPU resource release
- ✅ Blueprint registration (both root and `/v1` paths)

### Edge Cases and Error Handling
- ✅ No workers available scenarios
- ✅ Worker failure handling
- ✅ Invalid request data
- ✅ Network error handling
- ✅ Partial worker failures
- ✅ Malformed JSON requests

### Concurrency and Thread Safety
- ✅ Thread-safe queue management
- ✅ Concurrent request handling
- ✅ Lock-based synchronization
- ✅ Request counting accuracy

### Load Balancing
- ✅ Least busy worker selection
- ✅ Request distribution across workers
- ✅ Queue size tracking
- ✅ Worker load balancing under concurrent load

## Test Design Patterns

### Mocking Strategy
- **External HTTP requests** are mocked using `unittest.mock.patch`
- **Worker responses** are simulated with configurable mock objects
- **Network failures** are simulated by raising exceptions

### Fixtures and Setup
- **`reset_balancer_state`** fixture ensures clean state between tests
- **`client`** fixture provides Flask test client
- **Setup/teardown** methods reset global variables

### Parametrized Tests
Tests use pytest parametrization to test multiple scenarios:
```python
@pytest.mark.parametrize("worker_loads,expected_worker", [
    ({"http://worker1:8000": 0, "http://worker2:8000": 5}, "http://worker1:8000"),
    ({"http://worker1:8000": 3, "http://worker2:8000": 1}, "http://worker2:8000"),
])
def test_least_busy_worker_selection(worker_loads, expected_worker):
    # Test implementation
```

## Adding New Tests

### For New Endpoints
1. Add endpoint constant to test imports
2. Create test class for the endpoint
3. Test success, failure, and edge cases
4. Add integration tests if needed

### For New Features
1. Add unit tests for individual functions
2. Add integration tests for complete workflows
3. Consider concurrency implications
4. Test error conditions

### Test Naming Convention
- Test methods: `test_<functionality>_<scenario>`
- Test classes: `Test<ComponentName>`
- Integration tests: Include "integration" in name or use `@pytest.mark.integration`

## Example Test Structure

```python
class TestNewFeature:
    """Test the new feature functionality."""
    
    def test_feature_success(self, client, reset_balancer_state):
        """Test successful feature operation."""
        # Setup
        # Exercise
        # Verify
        
    def test_feature_failure(self, client, reset_balancer_state):
        """Test feature failure handling."""
        # Setup failure condition
        # Exercise
        # Verify proper error handling
        
    @patch('gllm.balancer.requests.post')
    def test_feature_with_mocked_worker(self, mock_post, client, reset_balancer_state):
        """Test feature with mocked worker interactions."""
        # Setup mock
        # Exercise
        # Verify mock was called correctly
```

## Debugging Tests

### Running Tests with Debug Output
```bash
# Run with detailed output
pytest -v -s tests/

# Run with Python debugger on failure
pytest --pdb tests/

# Run single test with full output
pytest -v -s tests/test_balancer_pytest.py::TestWorkerManagement::test_add_worker_success
```

### Common Issues
1. **Global state pollution** - Ensure `reset_balancer_state` fixture is used
2. **Mock setup** - Verify mocks are properly configured and reset between tests
3. **Concurrency issues** - Use appropriate delays in concurrent tests

## Performance Considerations

- **Fast tests** (< 1 second) should be the majority
- **Slow tests** (> 1 second) should be marked with `@pytest.mark.slow`
- **Integration tests** may take longer but should still be reasonable
- **Concurrent tests** use small delays to simulate real timing

## CI/CD Integration

The tests are designed to run reliably in CI environments:
- No external dependencies required
- Deterministic behavior through mocking
- Appropriate timeouts for concurrent tests
- Clear test markers for selective running 