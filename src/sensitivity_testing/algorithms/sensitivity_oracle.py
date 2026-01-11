"""
Sensitivity Oracle with PAC Learning Bounds.

This module implements test prioritization based on sensitivity scores,
with theoretical guarantees derived from PAC (Probably Approximately Correct)
learning theory.

Key Insight: We can bound the probability of missing bugs given:
1. Sensitivity score distribution
2. Number of tests executed
3. Desired confidence level

This provides principled answers to "how much testing is enough?"

Reference:
    Valiant, L.G. (1984). "A theory of the learnable."
    Communications of the ACM, 27(11), 1134-1142.
    
    Walkinshaw, N. et al. (2024). "Bounding random test set size with 
    computational learning theory." FSE 2024.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
import heapq

logger = logging.getLogger(__name__)


@dataclass
class PrioritizationResult:
    """Result of test prioritization."""
    
    prioritized_tests: List[Tuple[float, Any]]
    """Tests sorted by priority (score, test)."""
    
    expected_bugs: float
    """Expected number of bugs given budget."""
    
    coverage_bound: float
    """Lower bound on coverage achieved."""
    
    required_additional_tests: int
    """Additional tests needed for target coverage."""


@dataclass
class CoverageBound:
    """PAC learning bound on coverage."""
    
    lower_bound: float
    """Lower bound on true coverage (with confidence)."""
    
    confidence: float
    """Confidence level of the bound."""
    
    sample_size: int
    """Number of samples used."""
    
    vc_dimension: Optional[int]
    """VC dimension if applicable."""


class SensitivityOracle:
    """
    Provides test prioritization with PAC learning bounds.
    
    This oracle:
    1. Prioritizes tests by sensitivity score (high sensitivity = likely buggy)
    2. Provides theoretical bounds on bug-finding probability
    3. Estimates required test budget for desired coverage
    
    The key theorem underlying this work:
    
    For a hypothesis class H with VC dimension d, to achieve error ≤ ε
    with probability ≥ 1-δ, we need:
    
        n ≥ (4/ε) * (d*log(2/ε) + log(2/δ))
    
    Example:
        >>> oracle = SensitivityOracle(epsilon=0.05, delta=0.01)
        >>> budget = oracle.required_tests(sensitivity_scores, target_coverage=0.95)
        >>> print(f"Need {budget} tests for 95% coverage")
    """
    
    def __init__(
        self,
        epsilon: float = 0.05,
        delta: float = 0.01,
        hypothesis_class: str = 'vc_dimension_based',
        sensitivity_weight: float = 2.0
    ):
        """
        Initialize sensitivity oracle.
        
        Args:
            epsilon: Error tolerance (how much coverage we can miss)
            delta: Confidence parameter (probability of bound failing)
            hypothesis_class: Type of hypothesis class for bounds
            sensitivity_weight: Weight given to sensitivity in prioritization
        """
        self.epsilon = epsilon
        self.delta = delta
        self.hypothesis_class = hypothesis_class
        self.sensitivity_weight = sensitivity_weight
        
        # Estimated VC dimension for compiler bug hypothesis class
        # This is an empirical estimate based on typical compiler complexity
        self._vc_dimension = 20
        
        logger.debug(f"Initialized SensitivityOracle with ε={epsilon}, δ={delta}")
    
    def prioritize(
        self,
        test_candidates: List[Any],
        sensitivity_scores: List[float],
        coverage_scores: Optional[List[float]] = None,
        diversity_scores: Optional[List[float]] = None
    ) -> PrioritizationResult:
        """
        Prioritize tests based on sensitivity and other factors.
        
        Uses a weighted combination of:
        - Sensitivity score (from Lyapunov analysis)
        - Coverage potential (if available)
        - Diversity (to avoid redundant tests)
        
        Args:
            test_candidates: List of test programs/inputs
            sensitivity_scores: Lyapunov-based sensitivity scores
            coverage_scores: Optional coverage potential scores
            diversity_scores: Optional diversity/novelty scores
            
        Returns:
            PrioritizationResult with sorted tests and bounds
        """
        if len(test_candidates) != len(sensitivity_scores):
            raise ValueError("Mismatched lengths: tests and scores")
        
        n = len(test_candidates)
        
        # Normalize scores to [0, 1]
        sens_normalized = self._normalize(sensitivity_scores)
        
        if coverage_scores is not None:
            cov_normalized = self._normalize(coverage_scores)
        else:
            cov_normalized = np.ones(n) * 0.5
        
        if diversity_scores is not None:
            div_normalized = self._normalize(diversity_scores)
        else:
            div_normalized = np.ones(n) * 0.5
        
        # Compute combined priority scores
        # Higher sensitivity gets more weight
        priority_scores = (
            self.sensitivity_weight * sens_normalized +
            1.0 * cov_normalized +
            0.5 * div_normalized
        )
        
        # Sort by priority (descending)
        sorted_indices = np.argsort(-priority_scores)
        prioritized = [
            (float(priority_scores[i]), test_candidates[i])
            for i in sorted_indices
        ]
        
        # Estimate expected bugs in top tests
        expected_bugs = self._estimate_expected_bugs(
            sensitivity_scores, sorted_indices
        )
        
        # Compute coverage bound
        coverage_bound = self.compute_coverage_bound(n).lower_bound
        
        # Compute additional tests needed for target
        required_additional = max(0, 
            self.required_tests(sensitivity_scores, target_coverage=0.95) - n
        )
        
        return PrioritizationResult(
            prioritized_tests=prioritized,
            expected_bugs=expected_bugs,
            coverage_bound=coverage_bound,
            required_additional_tests=required_additional
        )
    
    def required_tests(
        self,
        sensitivity_scores: List[float],
        target_coverage: float = 0.95
    ) -> int:
        """
        Calculate required number of tests for target coverage.
        
        Uses PAC learning bounds:
        n ≥ (4/ε) * (d*log(2/ε) + log(2/δ))
        
        where ε = 1 - target_coverage
        
        Args:
            sensitivity_scores: Observed sensitivity distribution
            target_coverage: Desired coverage (0-1)
            
        Returns:
            Minimum number of tests required
        """
        # Effective epsilon based on target coverage
        epsilon = 1 - target_coverage
        
        if epsilon <= 0:
            return float('inf')
        
        # Standard PAC bound
        d = self._vc_dimension
        
        # n ≥ (4/ε) * (d*log(2/ε) + log(2/δ))
        basic_bound = (4.0 / epsilon) * (
            d * np.log(2.0 / epsilon) + np.log(2.0 / self.delta)
        )
        
        # Adjust based on sensitivity distribution
        # High sensitivity variance suggests more bugs to find
        if len(sensitivity_scores) > 1:
            sens_std = np.std(sensitivity_scores)
            sens_max = np.max(sensitivity_scores)
            
            # Increase budget if high sensitivity observed
            sensitivity_factor = 1 + sens_std + max(0, sens_max - 0.5)
            adjusted_bound = basic_bound * sensitivity_factor
        else:
            adjusted_bound = basic_bound
        
        return int(np.ceil(adjusted_bound))
    
    def compute_coverage_bound(
        self,
        num_tests: int,
        observed_bugs: int = 0
    ) -> CoverageBound:
        """
        Compute PAC bound on achieved coverage.
        
        Given n tests and k observed bugs, compute a lower bound
        on the coverage achieved with confidence 1-δ.
        
        Args:
            num_tests: Number of tests executed
            observed_bugs: Number of bugs found
            
        Returns:
            CoverageBound with lower bound and confidence
        """
        n = num_tests
        d = self._vc_dimension
        
        if n == 0:
            return CoverageBound(
                lower_bound=0.0,
                confidence=1 - self.delta,
                sample_size=0,
                vc_dimension=d
            )
        
        # Solve for ε: n = (4/ε) * (d*log(2/ε) + log(2/δ))
        # Lower bound on coverage = 1 - ε
        
        def bound_equation(eps):
            if eps <= 0 or eps >= 1:
                return float('inf')
            required = (4.0 / eps) * (d * np.log(2.0 / eps) + np.log(2.0 / self.delta))
            return abs(required - n)
        
        # Find epsilon that matches our sample size
        result = minimize_scalar(bound_equation, bounds=(0.001, 0.999), method='bounded')
        epsilon = result.x
        
        # Coverage bound
        lower_bound = max(0, 1 - epsilon)
        
        # Adjust for observed bugs (empirical Bayes)
        if observed_bugs > 0 and n > observed_bugs:
            # Bug rate suggests higher coverage needed
            bug_rate = observed_bugs / n
            adjusted_bound = lower_bound * (1 - bug_rate)
        else:
            adjusted_bound = lower_bound
        
        return CoverageBound(
            lower_bound=float(adjusted_bound),
            confidence=float(1 - self.delta),
            sample_size=n,
            vc_dimension=d
        )
    
    def estimate_bug_probability(
        self,
        sensitivity_score: float,
        prior_bugs_found: int = 0,
        prior_tests: int = 0
    ) -> float:
        """
        Estimate probability that a region contains bugs.
        
        Uses Bayesian inference with sensitivity as a feature:
        P(bug | sensitivity) ∝ P(sensitivity | bug) * P(bug)
        
        Args:
            sensitivity_score: Lyapunov-based sensitivity
            prior_bugs_found: Bugs found in similar regions
            prior_tests: Tests run in similar regions
            
        Returns:
            Estimated probability of bug presence
        """
        # Base rate (prior probability of bug)
        if prior_tests > 0:
            base_rate = (prior_bugs_found + 1) / (prior_tests + 2)  # Laplace smoothing
        else:
            base_rate = 0.01  # Default prior
        
        # Likelihood ratio based on sensitivity
        # High sensitivity (λ > 0) increases bug probability
        # Based on empirical observation that bugs cluster in chaotic regions
        
        if sensitivity_score > 0.5:
            likelihood_ratio = 10.0  # Strong evidence
        elif sensitivity_score > 0.1:
            likelihood_ratio = 3.0   # Moderate evidence
        elif sensitivity_score > 0:
            likelihood_ratio = 1.5   # Weak evidence
        else:
            likelihood_ratio = 0.5   # Evidence against
        
        # Bayesian update
        prior_odds = base_rate / (1 - base_rate)
        posterior_odds = prior_odds * likelihood_ratio
        posterior_prob = posterior_odds / (1 + posterior_odds)
        
        return float(np.clip(posterior_prob, 0, 1))
    
    def estimate_remaining_bugs(
        self,
        bugs_found: List[Dict],
        tests_run: int,
        sensitivity_landscape: 'SensitivityLandscape' = None
    ) -> Tuple[float, float]:
        """
        Estimate remaining undiscovered bugs using capture-recapture.
        
        Based on Böhme & Falk (2020) methodology.
        
        Args:
            bugs_found: List of bugs found so far
            tests_run: Number of tests executed
            sensitivity_landscape: Optional sensitivity map
            
        Returns:
            Tuple of (expected_remaining_bugs, 95%_confidence_interval)
        """
        n_bugs = len(bugs_found)
        
        if n_bugs < 2:
            # Not enough data for reliable estimate
            return (float('inf'), float('inf'))
        
        # Chao1 estimator (capture-recapture)
        # Count bugs found multiple times vs once
        bug_counts = {}
        for bug in bugs_found:
            bug_id = bug.get('id', str(bug))
            bug_counts[bug_id] = bug_counts.get(bug_id, 0) + 1
        
        f1 = sum(1 for c in bug_counts.values() if c == 1)  # Singletons
        f2 = sum(1 for c in bug_counts.values() if c == 2)  # Doubletons
        
        if f2 == 0:
            f2 = 1  # Avoid division by zero
        
        # Chao1 estimate
        estimated_total = n_bugs + (f1 * (f1 - 1)) / (2 * (f2 + 1))
        
        # Confidence interval (using asymptotic variance)
        var_chao = f2 * ((f1/f2)**4 / 4 + (f1/f2)**3 + (f1/f2)**2 / 2)
        ci_width = 1.96 * np.sqrt(var_chao)
        
        remaining = max(0, estimated_total - n_bugs)
        
        return (float(remaining), float(ci_width))
    
    def adaptive_budget_allocation(
        self,
        sensitivity_scores: List[float],
        total_budget: int,
        num_regions: int = 10
    ) -> Dict[int, int]:
        """
        Allocate test budget across sensitivity regions.
        
        Higher sensitivity regions get more tests, following
        Thompson sampling strategy.
        
        Args:
            sensitivity_scores: Scores for available tests
            total_budget: Total number of tests to allocate
            num_regions: Number of regions to partition
            
        Returns:
            Dict mapping region index to allocated budget
        """
        # Partition tests into regions by sensitivity
        scores = np.array(sensitivity_scores)
        percentiles = np.percentile(scores, np.linspace(0, 100, num_regions + 1))
        
        # Assign tests to regions
        regions = np.digitize(scores, percentiles[:-1]) - 1
        regions = np.clip(regions, 0, num_regions - 1)
        
        # Initial allocation proportional to sensitivity^2
        region_weights = np.zeros(num_regions)
        for i in range(num_regions):
            mask = regions == i
            if np.any(mask):
                region_weights[i] = np.mean(scores[mask]) ** 2
        
        # Normalize to get allocation
        if np.sum(region_weights) > 0:
            allocation = (region_weights / np.sum(region_weights)) * total_budget
        else:
            allocation = np.ones(num_regions) * (total_budget / num_regions)
        
        # Round to integers while preserving total
        int_allocation = np.floor(allocation).astype(int)
        remainder = total_budget - np.sum(int_allocation)
        
        # Distribute remainder to highest-sensitivity regions
        for _ in range(int(remainder)):
            int_allocation[np.argmax(region_weights)] += 1
        
        return {i: int(int_allocation[i]) for i in range(num_regions)}
    
    def _normalize(self, scores: List[float]) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        arr = np.array(scores)
        
        if len(arr) == 0:
            return arr
        
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        if max_val - min_val < 1e-10:
            return np.ones_like(arr) * 0.5
        
        return (arr - min_val) / (max_val - min_val)
    
    def _estimate_expected_bugs(
        self,
        sensitivity_scores: List[float],
        sorted_indices: np.ndarray,
        top_fraction: float = 0.2
    ) -> float:
        """Estimate expected bugs in top-ranked tests."""
        scores = np.array(sensitivity_scores)
        n = len(scores)
        
        if n == 0:
            return 0.0
        
        # Focus on top fraction
        top_k = max(1, int(n * top_fraction))
        top_indices = sorted_indices[:top_k]
        top_scores = scores[top_indices]
        
        # Sum of bug probabilities
        expected = sum(
            self.estimate_bug_probability(s) for s in top_scores
        )
        
        return float(expected)


def compute_pac_bound(
    num_samples: int,
    confidence: float = 0.99,
    vc_dimension: int = 20
) -> float:
    """
    Compute PAC learning bound on coverage.
    
    Args:
        num_samples: Number of test samples
        confidence: Desired confidence (1 - δ)
        vc_dimension: VC dimension of hypothesis class
        
    Returns:
        Lower bound on achieved coverage
    """
    oracle = SensitivityOracle(delta=1-confidence)
    oracle._vc_dimension = vc_dimension
    bound = oracle.compute_coverage_bound(num_samples)
    return bound.lower_bound
