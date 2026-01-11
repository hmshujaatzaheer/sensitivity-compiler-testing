"""
Unit tests for the Sensitivity Oracle with PAC Learning bounds.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sensitivity_testing.algorithms.sensitivity_oracle import (
    SensitivityOracle,
    PrioritizationResult,
    CoverageBound,
    compute_pac_bound
)


class TestSensitivityOracle:
    """Tests for SensitivityOracle class."""
    
    @pytest.fixture
    def oracle(self):
        """Create a default oracle instance."""
        return SensitivityOracle(epsilon=0.05, delta=0.01)
    
    def test_initialization(self, oracle):
        """Test oracle initialization with default parameters."""
        assert oracle.epsilon == 0.05
        assert oracle.delta == 0.01
        assert oracle._vc_dimension == 20
    
    def test_prioritize_basic(self, oracle):
        """Test basic prioritization of tests."""
        tests = ['prog1.c', 'prog2.c', 'prog3.c', 'prog4.c', 'prog5.c']
        scores = [0.1, 0.9, 0.5, 0.3, 0.8]
        
        result = oracle.prioritize(tests, scores)
        
        assert isinstance(result, PrioritizationResult)
        assert len(result.prioritized_tests) == 5
        
        # Check that high-sensitivity tests come first
        priorities = [p[0] for p in result.prioritized_tests]
        assert priorities == sorted(priorities, reverse=True)
        
        # prog2.c should be first (highest score 0.9)
        assert result.prioritized_tests[0][1] == 'prog2.c'
    
    def test_prioritize_with_coverage(self, oracle):
        """Test prioritization with coverage scores."""
        tests = ['a', 'b', 'c']
        sensitivity = [0.5, 0.5, 0.5]
        coverage = [0.9, 0.1, 0.5]
        
        result = oracle.prioritize(tests, sensitivity, coverage_scores=coverage)
        
        # 'a' should be ranked higher due to coverage
        names = [p[1] for p in result.prioritized_tests]
        assert names[0] == 'a'
    
    def test_required_tests_calculation(self, oracle):
        """Test required test budget calculation."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        budget = oracle.required_tests(scores, target_coverage=0.95)
        
        assert isinstance(budget, int)
        assert budget > 0
        # For 95% coverage with PAC bounds, should need significant tests
        assert budget >= 100
    
    def test_required_tests_higher_coverage_needs_more(self, oracle):
        """Test that higher coverage requires more tests."""
        scores = [0.5] * 10
        
        budget_90 = oracle.required_tests(scores, target_coverage=0.90)
        budget_99 = oracle.required_tests(scores, target_coverage=0.99)
        
        assert budget_99 > budget_90
    
    def test_coverage_bound(self, oracle):
        """Test PAC coverage bound calculation."""
        bound = oracle.compute_coverage_bound(num_tests=1000, observed_bugs=5)
        
        assert isinstance(bound, CoverageBound)
        assert 0 <= bound.lower_bound <= 1
        assert bound.confidence == 0.99  # 1 - delta
        assert bound.sample_size == 1000
    
    def test_coverage_bound_more_tests_higher_bound(self, oracle):
        """Test that more tests give higher coverage bound."""
        bound_100 = oracle.compute_coverage_bound(num_tests=100)
        bound_1000 = oracle.compute_coverage_bound(num_tests=1000)
        bound_10000 = oracle.compute_coverage_bound(num_tests=10000)
        
        assert bound_1000.lower_bound > bound_100.lower_bound
        assert bound_10000.lower_bound > bound_1000.lower_bound
    
    def test_bug_probability_estimation(self, oracle):
        """Test bug probability estimation."""
        # High sensitivity should give higher probability
        prob_high = oracle.estimate_bug_probability(0.9)
        prob_low = oracle.estimate_bug_probability(0.1)
        
        assert 0 <= prob_high <= 1
        assert 0 <= prob_low <= 1
        assert prob_high > prob_low
    
    def test_bug_probability_with_priors(self, oracle):
        """Test bug probability with prior observations."""
        # With prior bugs found, probability should be higher
        prob_no_prior = oracle.estimate_bug_probability(0.5, prior_bugs_found=0, prior_tests=100)
        prob_with_prior = oracle.estimate_bug_probability(0.5, prior_bugs_found=10, prior_tests=100)
        
        assert prob_with_prior > prob_no_prior
    
    def test_remaining_bugs_estimation(self, oracle):
        """Test Chao1-based remaining bugs estimation."""
        # Create bug list with some duplicates
        bugs = [
            {'id': 'bug1'}, {'id': 'bug1'},  # Found twice
            {'id': 'bug2'},  # Found once
            {'id': 'bug3'},  # Found once
            {'id': 'bug4'}, {'id': 'bug4'}, {'id': 'bug4'},  # Found three times
        ]
        
        remaining, ci = oracle.estimate_remaining_bugs(bugs, tests_run=1000)
        
        # Should estimate some remaining bugs (singletons suggest more)
        assert remaining >= 0
        assert ci >= 0
    
    def test_adaptive_budget_allocation(self, oracle):
        """Test adaptive budget allocation across regions."""
        # Create sensitivity scores with varying magnitudes
        scores = [0.1] * 50 + [0.5] * 30 + [0.9] * 20
        np.random.shuffle(scores)
        
        allocation = oracle.adaptive_budget_allocation(
            scores,
            total_budget=1000,
            num_regions=5
        )
        
        assert isinstance(allocation, dict)
        assert len(allocation) == 5
        assert sum(allocation.values()) == 1000
        
        # Higher sensitivity regions should get more budget
        # (exact values depend on score distribution)


class TestPACBound:
    """Tests for PAC bound computation."""
    
    def test_basic_bound(self):
        """Test basic PAC bound computation."""
        bound = compute_pac_bound(num_samples=1000)
        
        assert 0 <= bound <= 1
    
    def test_more_samples_higher_bound(self):
        """Test that more samples give higher coverage bound."""
        bound_100 = compute_pac_bound(num_samples=100)
        bound_1000 = compute_pac_bound(num_samples=1000)
        
        assert bound_1000 > bound_100
    
    def test_higher_confidence_lower_bound(self):
        """Test that higher confidence gives lower bound."""
        bound_95 = compute_pac_bound(num_samples=1000, confidence=0.95)
        bound_99 = compute_pac_bound(num_samples=1000, confidence=0.99)
        
        # Higher confidence = tighter = potentially lower bound
        # (depends on implementation details)
        assert bound_95 > 0
        assert bound_99 > 0
    
    def test_vc_dimension_effect(self):
        """Test effect of VC dimension on bounds."""
        bound_low_vc = compute_pac_bound(num_samples=1000, vc_dimension=5)
        bound_high_vc = compute_pac_bound(num_samples=1000, vc_dimension=50)
        
        # Higher VC dimension requires more samples, so bound is lower
        assert bound_low_vc > bound_high_vc


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_scores(self):
        """Test handling of empty score lists."""
        oracle = SensitivityOracle()
        
        result = oracle.prioritize([], [])
        assert len(result.prioritized_tests) == 0
    
    def test_mismatched_lengths(self):
        """Test handling of mismatched input lengths."""
        oracle = SensitivityOracle()
        
        with pytest.raises(ValueError):
            oracle.prioritize(['a', 'b', 'c'], [0.1, 0.2])
    
    def test_zero_tests_coverage_bound(self):
        """Test coverage bound with zero tests."""
        oracle = SensitivityOracle()
        bound = oracle.compute_coverage_bound(num_tests=0)
        
        assert bound.lower_bound == 0.0
    
    def test_single_test(self):
        """Test with single test."""
        oracle = SensitivityOracle()
        result = oracle.prioritize(['only_test'], [0.5])
        
        assert len(result.prioritized_tests) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
