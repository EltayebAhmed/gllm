import pytest
import sys
import os

# Add the gllm package to the path so we can import it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure pytest to show longer diffs
pytest.register_assert_rewrite('gllm.balancer')

def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "slow: mark test as slow running") 