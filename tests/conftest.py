"""
Pytest configuration and fixtures for sensitivity testing tests.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_trace():
    """Generate a sample execution trace."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def chaotic_trace():
    """Generate a trace from a chaotic system (logistic map)."""
    def logistic_map(x, r=4.0):
        return r * x * (1 - x)
    
    x = 0.1
    trace = []
    for _ in range(500):
        x = logistic_map(x)
        trace.append(x)
    
    return np.array(trace)


@pytest.fixture
def stable_trace():
    """Generate a trace from a stable system (damped oscillator)."""
    t = np.linspace(0, 10, 500)
    return np.exp(-0.5 * t) * np.sin(5 * t)


@pytest.fixture
def sample_program_source():
    """Return sample C program source code."""
    return '''
#include <stdio.h>

int main() {
    int sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += i;
    }
    printf("%d\\n", sum);
    return 0;
}
'''


@pytest.fixture
def sample_c_file(temp_dir, sample_program_source):
    """Create a sample C file in temp directory."""
    path = temp_dir / "test.c"
    path.write_text(sample_program_source)
    return path
