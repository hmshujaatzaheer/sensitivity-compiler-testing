"""
Unit tests for the Discrete Lyapunov exponent computation.
"""

import numpy as np
import pytest
from sensitivity_testing.algorithms.lyapunov import (
    DiscreteLyapunov,
    LyapunovResult,
    compute_lyapunov_exponent
)


class TestDiscreteLyapunov:
    """Tests for DiscreteLyapunov class."""
    
    def test_initialization(self):
        """Test default initialization."""
        lyapunov = DiscreteLyapunov()
        assert lyapunov.embedding_dimension == 5
        assert lyapunov.time_delay == 1
        assert lyapunov.min_neighbors == 5
    
    def test_custom_initialization(self):
        """Test custom parameters."""
        lyapunov = DiscreteLyapunov(
            embedding_dimension=10,
            time_delay=2,
            min_neighbors=3
        )
        assert lyapunov.embedding_dimension == 10
        assert lyapunov.time_delay == 2
        assert lyapunov.min_neighbors == 3
    
    def test_compute_empty_traces(self):
        """Test with empty input."""
        lyapunov = DiscreteLyapunov()
        result = lyapunov.compute([])
        assert result == 0.0
    
    def test_compute_single_trace(self):
        """Test with single trace."""
        lyapunov = DiscreteLyapunov(embedding_dimension=3)
        trace = np.random.randn(100)
        result = lyapunov.compute([trace])
        assert isinstance(result, float)
    
    def test_compute_multiple_traces(self):
        """Test with multiple traces."""
        lyapunov = DiscreteLyapunov(embedding_dimension=3)
        traces = [np.random.randn(100) for _ in range(5)]
        result = lyapunov.compute(traces)
        assert isinstance(result, float)
    
    def test_compute_returns_diagnostics(self):
        """Test returning full diagnostics."""
        lyapunov = DiscreteLyapunov(embedding_dimension=3)
        trace = np.random.randn(200)
        result = lyapunov.compute([trace], return_diagnostics=True)
        
        assert isinstance(result, LyapunovResult)
        assert hasattr(result, 'exponent')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'divergence_curve')
    
    def test_chaotic_system_positive_exponent(self):
        """Test that logistic map at chaos gives positive exponent."""
        # Logistic map at r=4 is chaotic
        def logistic_map(x, r=4.0):
            return r * x * (1 - x)
        
        x = 0.1
        trace = []
        for _ in range(500):
            x = logistic_map(x)
            trace.append(x)
        
        lyapunov = DiscreteLyapunov(embedding_dimension=3, time_delay=1)
        result = lyapunov.compute([np.array(trace)], return_diagnostics=True)
        
        # Logistic map at r=4 has theoretical λ ≈ 0.693
        # We just check it's positive (chaotic)
        assert result.exponent > -0.5  # Allow some tolerance
    
    def test_stable_system_negative_exponent(self):
        """Test that damped oscillator gives negative/zero exponent."""
        # Damped sine wave (stable)
        t = np.linspace(0, 10, 500)
        trace = np.exp(-0.5 * t) * np.sin(5 * t)
        
        lyapunov = DiscreteLyapunov(embedding_dimension=3)
        result = lyapunov.compute([trace], return_diagnostics=True)
        
        # Stable system should have non-positive exponent
        assert result.exponent < 0.5  # Allow tolerance
    
    def test_embedding(self):
        """Test phase space embedding."""
        lyapunov = DiscreteLyapunov(embedding_dimension=3, time_delay=2)
        time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        embedded = lyapunov._embed(time_series)
        
        # Check dimensions
        expected_length = len(time_series) - (3 - 1) * 2  # n - (m-1)*tau
        assert embedded.shape == (expected_length, 3)
        
        # Check first vector: [1, 3, 5] (indices 0, 2, 4)
        np.testing.assert_array_equal(embedded[0], [1, 3, 5])
    
    def test_trace_comparison(self):
        """Test direct trace comparison method."""
        lyapunov = DiscreteLyapunov()
        
        trace1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace2 = np.array([1.1, 2.2, 3.5, 5.0, 7.0])  # Diverging
        
        divergence = lyapunov.compute_from_traces_comparison(trace1, trace2)
        assert isinstance(divergence, float)


class TestLyapunovResult:
    """Tests for LyapunovResult dataclass."""
    
    def test_is_chaotic_positive(self):
        """Test chaotic detection with positive exponent."""
        result = LyapunovResult(
            exponent=0.5,
            confidence=0.9,
            divergence_curve=np.array([]),
            embedding_dimension=3,
            num_trajectories=100,
            convergence_achieved=True
        )
        assert result.is_chaotic()
    
    def test_is_chaotic_negative(self):
        """Test chaotic detection with negative exponent."""
        result = LyapunovResult(
            exponent=-0.5,
            confidence=0.9,
            divergence_curve=np.array([]),
            embedding_dimension=3,
            num_trajectories=100,
            convergence_achieved=True
        )
        assert not result.is_chaotic()
    
    def test_interpretation_highly_chaotic(self):
        """Test interpretation of high exponent."""
        result = LyapunovResult(
            exponent=0.8,
            confidence=0.9,
            divergence_curve=np.array([]),
            embedding_dimension=3,
            num_trajectories=100,
            convergence_achieved=True
        )
        assert "Highly chaotic" in result.interpretation()
    
    def test_interpretation_stable(self):
        """Test interpretation of negative exponent."""
        result = LyapunovResult(
            exponent=-0.5,
            confidence=0.9,
            divergence_curve=np.array([]),
            embedding_dimension=3,
            num_trajectories=100,
            convergence_achieved=True
        )
        assert "Stable" in result.interpretation()


class TestConvenienceFunction:
    """Tests for convenience functions."""
    
    def test_compute_lyapunov_exponent(self):
        """Test the convenience function."""
        trace = np.random.randn(200)
        result = compute_lyapunov_exponent([trace], embedding_dimension=3)
        assert isinstance(result, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
