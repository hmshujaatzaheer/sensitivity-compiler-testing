"""
Phase Transition Detection in Compiler Optimization Behavior.

This module implements algorithms for detecting critical parameter boundaries
where compiler optimization strategies qualitatively change. These phase
transitions indicate decision boundaries in the compiler—precisely where
edge cases and bugs tend to cluster.

Key Insight: Compilers make discrete decisions based on parameter thresholds
(loop trip counts, function sizes, array dimensions). At these thresholds,
behavior changes discontinuously—creating opportunities for bugs.

Methods implemented:
1. CUSUM (Cumulative Sum Control Chart) - O(n) online detection
2. PELT (Pruned Exact Linear Time) - O(n) offline detection
3. BOCPD (Bayesian Online Change Point Detection) - O(n²) probabilistic

Complexity: O(n) where n is parameter range size (for CUSUM/PELT)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of behavioral changes at phase transitions."""
    OPTIMIZATION_APPLIED = "optimization_applied"
    OPTIMIZATION_DISABLED = "optimization_disabled"
    CODE_PATH_SWITCH = "code_path_switch"
    VECTORIZATION_THRESHOLD = "vectorization_threshold"
    UNROLL_THRESHOLD = "unroll_threshold"
    INLINE_THRESHOLD = "inline_threshold"
    REGISTER_SPILL = "register_spill"
    UNKNOWN = "unknown"


@dataclass
class PhaseTransition:
    """Detected phase transition in compiler behavior."""
    
    parameter_name: str
    """Name of the parameter (e.g., 'loop_trip_count')."""
    
    parameter_value: float
    """Value at which transition occurs."""
    
    change_type: ChangeType
    """Type of behavioral change."""
    
    before_behavior: Dict
    """Behavior metrics before transition."""
    
    after_behavior: Dict
    """Behavior metrics after transition."""
    
    confidence: float
    """Confidence in detection (0-1)."""
    
    magnitude: float
    """Magnitude of behavioral change."""
    
    def __str__(self) -> str:
        return (f"PhaseTransition({self.parameter_name}={self.parameter_value}, "
                f"type={self.change_type.value}, confidence={self.confidence:.3f})")
    
    def to_dict(self) -> Dict:
        return {
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "change_type": self.change_type.value,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
            "before_behavior": self.before_behavior,
            "after_behavior": self.after_behavior
        }


class PhaseTransitionDetector:
    """
    Detects phase transitions in compiler optimization behavior.
    
    This class identifies critical parameter values where compiler behavior
    qualitatively changes. These transitions are valuable for testing because:
    1. Bugs often cluster at decision boundaries
    2. Edge cases naturally occur near thresholds
    3. Optimization switches create discontinuities
    
    Example:
        >>> detector = PhaseTransitionDetector(method='cusum')
        >>> transitions = detector.detect(
        ...     parameter_name='loop_trip_count',
        ...     parameter_range=range(1, 1000),
        ...     program_template='for(int i=0; i<{N}; i++) sum += a[i];'
        ... )
        >>> for t in transitions:
        ...     print(f"Transition at N={t.parameter_value}: {t.change_type}")
    """
    
    def __init__(
        self,
        method: str = 'cusum',
        significance_level: float = 0.05,
        min_segment_length: int = 5,
        sensitivity: float = 1.0
    ):
        """
        Initialize phase transition detector.
        
        Args:
            method: Detection method ('cusum', 'pelt', 'bocpd')
            significance_level: Statistical significance threshold
            min_segment_length: Minimum samples between transitions
            sensitivity: Sensitivity multiplier (higher = more detections)
        """
        self.method = method
        self.significance_level = significance_level
        self.min_segment_length = min_segment_length
        self.sensitivity = sensitivity
        
        # Method-specific parameters
        self._cusum_threshold = 4.0 / sensitivity
        self._pelt_penalty = 3.0 * np.log(100)  # BIC penalty
        
        logger.debug(f"Initialized PhaseTransitionDetector with method={method}")
    
    def detect(
        self,
        parameter_name: str,
        parameter_range: range,
        program_template: str,
        compiler_manager: 'CompilerManager' = None,
        behavior_metric: Callable = None
    ) -> List[PhaseTransition]:
        """
        Scan parameter space to detect phase transitions.
        
        Args:
            parameter_name: Name of parameter to vary
            parameter_range: Range of values to test
            program_template: Program template with {parameter_name} placeholder
            compiler_manager: CompilerManager for compilation
            behavior_metric: Function to compute behavior metric from compilation
            
        Returns:
            List of detected PhaseTransition objects
        """
        # Collect behavior data across parameter range
        behaviors = []
        
        for value in parameter_range:
            # Generate program with this parameter value
            program_source = program_template.replace(f'{{{parameter_name}}}', str(value))
            program_source = program_source.replace('{N}', str(value))
            
            # Get behavior metric
            if compiler_manager and behavior_metric:
                behavior = self._measure_behavior(
                    program_source, compiler_manager, behavior_metric
                )
            else:
                # Simulated behavior for testing
                behavior = self._simulate_behavior(value)
            
            behaviors.append({
                'value': value,
                'metric': behavior
            })
        
        # Extract time series of metrics
        values = np.array([b['value'] for b in behaviors])
        metrics = np.array([b['metric'] for b in behaviors])
        
        # Detect change points
        if self.method == 'cusum':
            change_points = self._detect_cusum(metrics)
        elif self.method == 'pelt':
            change_points = self._detect_pelt(metrics)
        elif self.method == 'bocpd':
            change_points = self._detect_bocpd(metrics)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Convert to PhaseTransition objects
        transitions = []
        for cp in change_points:
            if cp < len(values):
                transition = self._create_transition(
                    parameter_name, values, metrics, cp
                )
                if transition:
                    transitions.append(transition)
        
        logger.info(f"Detected {len(transitions)} phase transitions for {parameter_name}")
        return transitions
    
    def detect_from_traces(
        self,
        traces: Dict[str, 'ExecutionTrace']
    ) -> List[PhaseTransition]:
        """
        Detect phase transitions from collected execution traces.
        
        Args:
            traces: Dictionary mapping configuration to traces
            
        Returns:
            List of detected transitions
        """
        if not traces:
            return []
        
        # Extract optimization levels and metrics
        opt_levels = []
        metrics = []
        
        for key, trace in traces.items():
            # Parse optimization level from key (e.g., 'gcc_-O2')
            parts = key.split('_')
            if len(parts) >= 2:
                opt = parts[-1]
                opt_levels.append(opt)
                metrics.append(trace.execution_time_ms if hasattr(trace, 'execution_time_ms') else len(trace.to_vector()))
        
        if len(metrics) < 3:
            return []
        
        # Detect transitions in the metrics
        metrics_array = np.array(metrics)
        change_points = self._detect_cusum(metrics_array)
        
        transitions = []
        for cp in change_points:
            if cp < len(opt_levels):
                transitions.append(PhaseTransition(
                    parameter_name='optimization_level',
                    parameter_value=cp,
                    change_type=ChangeType.OPTIMIZATION_APPLIED,
                    before_behavior={'metric': float(metrics[max(0, cp-1)])},
                    after_behavior={'metric': float(metrics[min(cp, len(metrics)-1)])},
                    confidence=0.8,
                    magnitude=abs(metrics[cp] - metrics[max(0, cp-1)]) if cp > 0 else 0
                ))
        
        return transitions
    
    def _detect_cusum(self, data: np.ndarray) -> List[int]:
        """
        CUSUM (Cumulative Sum) change point detection.
        
        O(n) online detection algorithm based on cumulative deviations
        from the mean.
        
        Args:
            data: 1D array of observations
            
        Returns:
            List of change point indices
        """
        n = len(data)
        if n < 2 * self.min_segment_length:
            return []
        
        # Compute mean and standard deviation
        mean = np.mean(data)
        std = np.std(data)
        
        if std < 1e-10:
            return []  # No variation
        
        # Normalized CUSUM
        z = (data - mean) / std
        
        # Cumulative sums (positive and negative)
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        
        for i in range(1, n):
            s_pos[i] = max(0, s_pos[i-1] + z[i] - 0.5)
            s_neg[i] = max(0, s_neg[i-1] - z[i] - 0.5)
        
        # Find threshold crossings
        threshold = self._cusum_threshold
        change_points = []
        
        i = 0
        while i < n:
            if s_pos[i] > threshold or s_neg[i] > threshold:
                # Found a change point
                change_points.append(i)
                
                # Reset CUSUM
                s_pos[i:] = 0
                s_neg[i:] = 0
                
                # Skip minimum segment length
                i += self.min_segment_length
            else:
                i += 1
        
        return change_points
    
    def _detect_pelt(self, data: np.ndarray) -> List[int]:
        """
        PELT (Pruned Exact Linear Time) change point detection.
        
        O(n) offline detection using dynamic programming with pruning.
        Based on Killick et al. (2012).
        
        Args:
            data: 1D array of observations
            
        Returns:
            List of change point indices
        """
        n = len(data)
        if n < 2 * self.min_segment_length:
            return []
        
        # Cost function: negative log-likelihood for Gaussian
        def segment_cost(start: int, end: int) -> float:
            if end <= start:
                return np.inf
            segment = data[start:end]
            if len(segment) < 2:
                return 0
            var = np.var(segment)
            if var < 1e-10:
                return 0
            return len(segment) * np.log(var)
        
        # Dynamic programming with pruning
        penalty = self._pelt_penalty
        
        # F[t] = minimum cost for data[0:t]
        F = np.full(n + 1, np.inf)
        F[0] = -penalty
        
        # Track optimal previous change points
        prev_cp = [[] for _ in range(n + 1)]
        
        # Candidate change points (for pruning)
        candidates = {0}
        
        for t in range(self.min_segment_length, n + 1):
            # Find best previous change point
            best_cost = np.inf
            best_cp = 0
            
            prune_set = set()
            
            for s in candidates:
                if t - s >= self.min_segment_length:
                    cost = F[s] + segment_cost(s, t) + penalty
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_cp = s
                    
                    # Pruning: remove if can't be optimal
                    if cost > F[t]:
                        prune_set.add(s)
            
            F[t] = best_cost
            prev_cp[t] = prev_cp[best_cp] + ([best_cp] if best_cp > 0 else [])
            
            # Update candidates
            candidates -= prune_set
            candidates.add(t - 1)
        
        return prev_cp[n]
    
    def _detect_bocpd(self, data: np.ndarray) -> List[int]:
        """
        BOCPD (Bayesian Online Change Point Detection).
        
        O(n²) probabilistic detection using Bayesian inference.
        Based on Adams & MacKay (2007).
        
        Args:
            data: 1D array of observations
            
        Returns:
            List of change point indices
        """
        n = len(data)
        if n < 2 * self.min_segment_length:
            return []
        
        # Hyperparameters for Normal-Gamma prior
        mu0 = np.mean(data)
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = 1.0
        
        # Hazard rate (probability of change point)
        hazard = 1.0 / 250  # Expected run length of 250
        
        # Run length probabilities
        # R[t, r] = P(run length = r at time t)
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0
        
        # Sufficient statistics for each run length
        sum_x = np.zeros(n + 1)
        sum_x2 = np.zeros(n + 1)
        counts = np.zeros(n + 1)
        
        # Most probable run lengths
        max_run_length = np.zeros(n)
        
        for t in range(1, n + 1):
            x = data[t - 1]
            
            # Update sufficient statistics
            new_sum_x = sum_x + x
            new_sum_x2 = sum_x2 + x * x
            new_counts = counts + 1
            
            # Compute predictive probabilities
            pred_probs = np.zeros(t)
            
            for r in range(t):
                # Posterior parameters
                kappa = kappa0 + new_counts[r]
                mu = (kappa0 * mu0 + new_sum_x[r]) / kappa
                alpha = alpha0 + new_counts[r] / 2
                beta = beta0 + 0.5 * (new_sum_x2[r] - new_sum_x[r]**2 / max(new_counts[r], 1))
                beta += kappa0 * new_counts[r] * (mu0 - new_sum_x[r] / max(new_counts[r], 1))**2 / (2 * kappa)
                
                # Student-t predictive probability
                df = 2 * alpha
                scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
                
                if scale > 0:
                    pred_probs[r] = stats.t.pdf((x - mu) / scale, df) / scale
                else:
                    pred_probs[r] = 1e-10
            
            # Growth probabilities (no change point)
            growth_probs = R[t-1, :t] * pred_probs * (1 - hazard)
            
            # Change point probability
            cp_prob = np.sum(R[t-1, :t] * pred_probs) * hazard
            
            # Update run length distribution
            R[t, 1:t+1] = growth_probs
            R[t, 0] = cp_prob
            
            # Normalize
            total = np.sum(R[t, :t+1])
            if total > 0:
                R[t, :t+1] /= total
            
            # Track most probable run length
            max_run_length[t-1] = np.argmax(R[t, :t+1])
            
            # Update sufficient statistics
            sum_x = np.roll(new_sum_x, 1)
            sum_x[0] = 0
            sum_x2 = np.roll(new_sum_x2, 1)
            sum_x2[0] = 0
            counts = np.roll(new_counts, 1)
            counts[0] = 0
        
        # Find change points where run length drops to 0
        change_points = []
        for t in range(self.min_segment_length, n - self.min_segment_length):
            if max_run_length[t] == 0 and R[t+1, 0] > 0.5:
                change_points.append(t)
        
        return change_points
    
    def _create_transition(
        self,
        parameter_name: str,
        values: np.ndarray,
        metrics: np.ndarray,
        change_point: int
    ) -> Optional[PhaseTransition]:
        """Create PhaseTransition object from detected change point."""
        if change_point <= 0 or change_point >= len(values) - 1:
            return None
        
        # Compute statistics before and after
        before_metrics = metrics[max(0, change_point - self.min_segment_length):change_point]
        after_metrics = metrics[change_point:min(len(metrics), change_point + self.min_segment_length)]
        
        if len(before_metrics) == 0 or len(after_metrics) == 0:
            return None
        
        before_mean = np.mean(before_metrics)
        after_mean = np.mean(after_metrics)
        
        # Compute magnitude of change
        magnitude = abs(after_mean - before_mean)
        
        # Compute confidence using t-test
        if len(before_metrics) > 1 and len(after_metrics) > 1:
            _, p_value = stats.ttest_ind(before_metrics, after_metrics)
            confidence = 1 - p_value
        else:
            confidence = 0.5
        
        # Classify change type
        change_type = self._classify_change(before_mean, after_mean, parameter_name)
        
        return PhaseTransition(
            parameter_name=parameter_name,
            parameter_value=float(values[change_point]),
            change_type=change_type,
            before_behavior={'mean': float(before_mean), 'std': float(np.std(before_metrics))},
            after_behavior={'mean': float(after_mean), 'std': float(np.std(after_metrics))},
            confidence=float(max(0, min(1, confidence))),
            magnitude=float(magnitude)
        )
    
    def _classify_change(
        self,
        before: float,
        after: float,
        parameter_name: str
    ) -> ChangeType:
        """Classify the type of behavioral change."""
        # Heuristics based on parameter name and change direction
        param_lower = parameter_name.lower()
        
        if 'loop' in param_lower or 'unroll' in param_lower:
            return ChangeType.UNROLL_THRESHOLD
        elif 'inline' in param_lower or 'function' in param_lower:
            return ChangeType.INLINE_THRESHOLD
        elif 'vector' in param_lower or 'simd' in param_lower:
            return ChangeType.VECTORIZATION_THRESHOLD
        elif 'register' in param_lower or 'spill' in param_lower:
            return ChangeType.REGISTER_SPILL
        elif after < before:
            return ChangeType.OPTIMIZATION_APPLIED
        else:
            return ChangeType.OPTIMIZATION_DISABLED
    
    def _measure_behavior(
        self,
        program_source: str,
        compiler_manager: 'CompilerManager',
        behavior_metric: Callable
    ) -> float:
        """Compile program and measure behavior metric."""
        # Create temporary program
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(program_source)
            temp_path = Path(f.name)
        
        try:
            # Compile with default settings
            from .core.program import TestProgram
            program = TestProgram(path=temp_path, source_code=program_source)
            result = compiler_manager.compile(program, 'gcc', '-O2')
            
            if result.success:
                return behavior_metric(result)
            else:
                return 0.0
        finally:
            temp_path.unlink(missing_ok=True)
    
    def _simulate_behavior(self, value: float) -> float:
        """Simulate compiler behavior for testing without actual compilation."""
        # Simulate typical compiler behavior with phase transitions
        
        # Unrolling threshold around 8
        if value < 8:
            base = 1.0
        else:
            base = 0.7  # Speedup from unrolling
        
        # Vectorization threshold around 64
        if value >= 64:
            base *= 0.5  # Speedup from vectorization
        
        # Add some noise
        noise = np.random.normal(0, 0.05)
        
        return base + noise


def detect_phase_transitions(
    parameter_range: range,
    behavior_function: Callable[[int], float],
    method: str = 'cusum'
) -> List[int]:
    """
    Convenience function to detect phase transitions.
    
    Args:
        parameter_range: Range of parameter values
        behavior_function: Function mapping parameter to behavior metric
        method: Detection method
        
    Returns:
        List of parameter values where transitions occur
    """
    detector = PhaseTransitionDetector(method=method)
    
    # Collect behavior data
    metrics = np.array([behavior_function(v) for v in parameter_range])
    
    # Detect change points
    if method == 'cusum':
        change_points = detector._detect_cusum(metrics)
    elif method == 'pelt':
        change_points = detector._detect_pelt(metrics)
    else:
        change_points = detector._detect_bocpd(metrics)
    
    # Convert indices to parameter values
    values = list(parameter_range)
    return [values[cp] for cp in change_points if cp < len(values)]
