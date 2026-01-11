"""
Integration tests for the sensitivity-theoretic compiler testing framework.

These tests verify end-to-end functionality of the framework components
working together.
"""

import pytest
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestFrameworkIntegration:
    """Integration tests for the main framework."""
    
    @pytest.fixture
    def sample_program(self):
        """Create a simple test program."""
        source = '''
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(source)
            return Path(f.name)
    
    def test_analyze_workflow(self, sample_program):
        """Test complete analysis workflow."""
        from sensitivity_testing.algorithms import DiscreteLyapunov, PhaseTransitionDetector
        import numpy as np
        
        # Simulate trace collection (actual compilation would require GCC)
        trace_data = np.random.randn(1000)
        
        # Test Lyapunov computation
        lyapunov = DiscreteLyapunov(embedding_dimension=3)
        result = lyapunov.compute(trace_data, return_diagnostics=True)
        
        assert hasattr(result, 'exponent')
        assert hasattr(result, 'confidence')
        assert isinstance(result.exponent, float)
    
    def test_phase_detection_workflow(self):
        """Test phase detection with synthetic data."""
        from sensitivity_testing.algorithms import PhaseTransitionDetector
        import numpy as np
        
        detector = PhaseTransitionDetector(method='cusum')
        
        # Create synthetic behavior with clear transition
        def behavior(n):
            base = 1.0 if n < 50 else 0.5
            return base + np.random.normal(0, 0.1)
        
        # This tests the internal methods
        data = np.array([behavior(i) for i in range(100)])
        transitions = detector._detect_cusum(data)
        
        # Should detect transition around n=50
        assert isinstance(transitions, list)
    
    def test_sensitivity_oracle_workflow(self):
        """Test sensitivity oracle prioritization."""
        from sensitivity_testing.algorithms import SensitivityOracle
        
        oracle = SensitivityOracle(epsilon=0.05, delta=0.01)
        
        # Test programs with varying sensitivity
        tests = ['prog1', 'prog2', 'prog3', 'prog4', 'prog5']
        scores = [0.1, 0.9, 0.5, 0.3, 0.8]
        
        result = oracle.prioritize(tests, scores)
        
        # Verify prioritization
        assert len(result.prioritized_tests) == 5
        # High sensitivity should come first
        assert result.prioritized_tests[0][1] == 'prog2'
    
    def test_coverage_bound_computation(self):
        """Test PAC coverage bound computation."""
        from sensitivity_testing.algorithms import SensitivityOracle
        
        oracle = SensitivityOracle()
        bound = oracle.compute_coverage_bound(num_tests=1000)
        
        assert 0 <= bound.lower_bound <= 1
        assert bound.confidence == 0.99  # 1 - delta
    
    def test_algorithm_pipeline(self):
        """Test algorithms work together in pipeline."""
        from sensitivity_testing.algorithms import (
            DiscreteLyapunov,
            PhaseTransitionDetector,
            SensitivityOracle
        )
        import numpy as np
        
        # Step 1: Generate synthetic traces
        traces = [np.random.randn(500) for _ in range(10)]
        
        # Step 2: Compute Lyapunov exponents
        lyapunov = DiscreteLyapunov(embedding_dimension=3)
        sensitivity_scores = []
        for trace in traces:
            score = lyapunov.compute(trace)
            sensitivity_scores.append(max(0, score))
        
        # Step 3: Prioritize using oracle
        oracle = SensitivityOracle()
        test_names = [f'test_{i}' for i in range(10)]
        result = oracle.prioritize(test_names, sensitivity_scores)
        
        # Verify pipeline output
        assert len(result.prioritized_tests) == 10
        assert result.expected_bugs >= 0


class TestOracleIntegration:
    """Integration tests for bug detection oracles."""
    
    def test_differential_oracle_detection(self):
        """Test differential oracle bug detection logic."""
        from sensitivity_testing.oracles.differential import DifferentialOracle, Bug
        
        oracle = DifferentialOracle()
        
        # Oracle should handle empty results
        result = oracle.check(None, {})
        assert result is None
    
    def test_crash_oracle_patterns(self):
        """Test crash oracle pattern detection."""
        from sensitivity_testing.oracles.crash import CrashOracle
        
        oracle = CrashOracle(check_sanitizers=True)
        
        # Verify sanitizer patterns are configured
        assert 'AddressSanitizer' in oracle.sanitizer_patterns
        assert 'UndefinedBehaviorSanitizer' in oracle.sanitizer_patterns


class TestConfigIntegration:
    """Test configuration system integration."""
    
    def test_framework_config_defaults(self):
        """Test default configuration values."""
        from sensitivity_testing.utils.config import FrameworkConfig
        
        config = FrameworkConfig()
        
        assert config.embedding_dimension == 5
        assert config.epsilon == 0.05
        assert config.delta == 0.01
    
    def test_config_save_load(self):
        """Test configuration serialization."""
        from sensitivity_testing.utils.config import FrameworkConfig
        import tempfile
        import json
        
        config = FrameworkConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = Path(f.name)
        
        config.save(path)
        
        with open(path) as f:
            data = json.load(f)
        
        assert data['embedding_dimension'] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
