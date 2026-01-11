"""
Discrete Lyapunov Exponent Computation for Compiler Execution Traces.

This module implements the Rosenstein et al. (1993) algorithm adapted for
discrete compiler execution traces. The Lyapunov exponent measures the
rate of separation of infinitesimally close trajectories, indicating
sensitivity to initial conditions (chaos).

Key insight: Compiler behavior can be modeled as a dynamical system where
input programs are initial conditions and execution traces are trajectories.
Regions with positive Lyapunov exponents exhibit chaotic behavior—small
input changes cause exponential divergence—and correlate with bug-prone areas.

Reference:
    Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993).
    "A practical method for calculating largest Lyapunov exponents from small data sets."
    Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
    
    Note: Rosenstein et al. describe their algorithm as "fast, easy to implement, 
    and robust" but provide no formal complexity analysis.

Complexity: O(T log T) where T is trace length
    This complexity is achieved through our use of k-d trees (scipy.spatial.KDTree) 
    for nearest-neighbor search. This is our implementation choice, not a claim 
    from the original Rosenstein paper.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy.spatial import KDTree
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class LyapunovResult:
    """Result of Lyapunov exponent computation."""
    
    exponent: float
    """The largest Lyapunov exponent (λ). λ > 0 indicates chaos."""
    
    confidence: float
    """Confidence in the estimate (R² of linear fit)."""
    
    divergence_curve: np.ndarray
    """Average logarithmic divergence over time."""
    
    embedding_dimension: int
    """Embedding dimension used."""
    
    num_trajectories: int
    """Number of trajectory pairs analyzed."""
    
    convergence_achieved: bool
    """Whether the estimate converged."""
    
    def is_chaotic(self, threshold: float = 0.0) -> bool:
        """Check if system exhibits chaotic behavior."""
        return self.exponent > threshold and self.confidence > 0.8
    
    def interpretation(self) -> str:
        """Get human-readable interpretation."""
        if self.exponent > 0.5:
            return "Highly chaotic (strong sensitivity to perturbations)"
        elif self.exponent > 0.1:
            return "Moderately chaotic (noticeable sensitivity)"
        elif self.exponent > 0:
            return "Weakly chaotic (edge of stability)"
        elif self.exponent > -0.1:
            return "Marginally stable (near phase transition)"
        else:
            return "Stable (robust to perturbations)"


class DiscreteLyapunov:
    """
    Computes discrete Lyapunov exponents from execution traces.
    
    The algorithm works by:
    1. Embedding the trace in a higher-dimensional phase space
    2. Finding nearest neighbors for each point
    3. Tracking how neighbor distances evolve over time
    4. Computing the average rate of divergence
    
    Example:
        >>> lyapunov = DiscreteLyapunov(embedding_dimension=5)
        >>> traces = [trace1, trace2, trace3]  # numpy arrays
        >>> result = lyapunov.compute(traces)
        >>> print(f"λ = {result.exponent:.4f}")
        >>> if result.is_chaotic():
        ...     print("Bug-prone region detected!")
    """
    
    def __init__(
        self,
        embedding_dimension: int = 5,
        time_delay: int = 1,
        min_neighbors: int = 5,
        max_iterations: int = 100,
        theiler_window: int = 10,
        convergence_threshold: float = 0.01
    ):
        """
        Initialize Lyapunov exponent calculator.
        
        Args:
            embedding_dimension: Dimension of reconstructed phase space (m).
                Higher values capture more complex dynamics but require more data.
            time_delay: Time delay for phase space reconstruction (τ).
                Should be chosen to minimize autocorrelation.
            min_neighbors: Minimum number of neighbors required for reliable estimate.
            max_iterations: Maximum time steps to track divergence.
            theiler_window: Minimum temporal separation to avoid correlated neighbors.
            convergence_threshold: Threshold for convergence detection.
        """
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        self.min_neighbors = min_neighbors
        self.max_iterations = max_iterations
        self.theiler_window = theiler_window
        self.convergence_threshold = convergence_threshold
        
        logger.debug(f"Initialized DiscreteLyapunov with m={embedding_dimension}, τ={time_delay}")
    
    def compute(
        self,
        traces: Union[List[np.ndarray], np.ndarray],
        return_diagnostics: bool = False
    ) -> Union[float, LyapunovResult]:
        """
        Compute the largest Lyapunov exponent from execution traces.
        
        This implements the Rosenstein et al. (1993) algorithm:
        1. Reconstruct phase space via time-delay embedding
        2. Find nearest neighbors (excluding temporally close points)
        3. Track divergence of neighbors over time
        4. Fit exponential growth to get Lyapunov exponent
        
        Args:
            traces: List of execution trace vectors or 2D array
            return_diagnostics: If True, return full LyapunovResult
            
        Returns:
            Lyapunov exponent (float) or LyapunovResult if return_diagnostics=True
        
        Complexity: O(T log T) where T is total trace length
        """
        # Convert to numpy array if needed
        if isinstance(traces, list):
            if len(traces) == 0:
                return 0.0 if not return_diagnostics else LyapunovResult(
                    exponent=0.0, confidence=0.0, divergence_curve=np.array([]),
                    embedding_dimension=self.embedding_dimension, num_trajectories=0,
                    convergence_achieved=False
                )
            traces = np.vstack(traces) if len(traces[0].shape) > 0 else np.array(traces)
        
        # Flatten if multi-dimensional
        if len(traces.shape) > 1:
            traces = traces.flatten()
        
        # Check minimum length
        min_length = (self.embedding_dimension - 1) * self.time_delay + self.max_iterations + 1
        if len(traces) < min_length:
            logger.warning(f"Trace too short ({len(traces)} < {min_length}). Padding with zeros.")
            traces = np.pad(traces, (0, min_length - len(traces)))
        
        # Step 1: Phase space reconstruction via time-delay embedding
        embedded = self._embed(traces)
        
        # Step 2: Build KD-tree for efficient nearest neighbor search (O(N log N))
        tree = KDTree(embedded)
        
        # Step 3: Find nearest neighbors and track divergence
        divergence = self._compute_divergence(embedded, tree)
        
        # Step 4: Estimate Lyapunov exponent via linear regression
        exponent, confidence, convergence = self._estimate_exponent(divergence)
        
        if return_diagnostics:
            return LyapunovResult(
                exponent=exponent,
                confidence=confidence,
                divergence_curve=divergence,
                embedding_dimension=self.embedding_dimension,
                num_trajectories=len(embedded),
                convergence_achieved=convergence
            )
        
        return exponent
    
    def _embed(self, time_series: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase space using time-delay embedding (Takens' theorem).
        
        Given a 1D time series x(t), construct m-dimensional vectors:
        X(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]
        
        Args:
            time_series: 1D array of observations
            
        Returns:
            2D array of shape (N, m) where N is number of embedded points
        """
        n = len(time_series)
        m = self.embedding_dimension
        tau = self.time_delay
        
        # Number of embedded vectors
        num_vectors = n - (m - 1) * tau
        
        if num_vectors <= 0:
            raise ValueError(f"Time series too short for embedding: {n} < {(m-1)*tau + 1}")
        
        # Construct embedded matrix
        embedded = np.zeros((num_vectors, m))
        for i in range(m):
            embedded[:, i] = time_series[i * tau : i * tau + num_vectors]
        
        return embedded
    
    def _compute_divergence(
        self,
        embedded: np.ndarray,
        tree: KDTree
    ) -> np.ndarray:
        """
        Compute average logarithmic divergence over time.
        
        For each point, find its nearest neighbor (excluding temporally
        close points), then track how their distance evolves.
        
        Args:
            embedded: Embedded phase space points
            tree: KD-tree for neighbor search
            
        Returns:
            Array of average log(divergence) at each time step
        """
        n = len(embedded)
        max_t = min(self.max_iterations, n // 4)
        
        # Storage for divergence at each time step
        divergence_sum = np.zeros(max_t)
        divergence_count = np.zeros(max_t)
        
        # Find nearest neighbors for each point
        # Query k+1 neighbors since first result is the point itself
        k = self.min_neighbors + 1
        distances, indices = tree.query(embedded, k=k)
        
        for i in range(n - max_t):
            # Find nearest neighbor outside Theiler window
            for j in range(1, k):
                neighbor_idx = indices[i, j]
                
                # Skip if too close in time (Theiler window)
                if abs(neighbor_idx - i) <= self.theiler_window:
                    continue
                
                # Skip if neighbor would go out of bounds
                if neighbor_idx + max_t >= n:
                    continue
                
                initial_dist = distances[i, j]
                if initial_dist < 1e-10:  # Avoid log(0)
                    continue
                
                # Track divergence over time
                for t in range(max_t):
                    dist_t = np.linalg.norm(embedded[i + t] - embedded[neighbor_idx + t])
                    if dist_t > 1e-10:
                        divergence_sum[t] += np.log(dist_t / initial_dist)
                        divergence_count[t] += 1
                
                break  # Only use closest valid neighbor
        
        # Average divergence at each time step
        with np.errstate(divide='ignore', invalid='ignore'):
            divergence = np.where(
                divergence_count > 0,
                divergence_sum / divergence_count,
                0.0
            )
        
        return divergence
    
    def _estimate_exponent(
        self,
        divergence: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Estimate Lyapunov exponent from divergence curve via linear regression.
        
        The Lyapunov exponent is the slope of the linear region of
        <ln(divergence)> vs time.
        
        Args:
            divergence: Average log divergence at each time step
            
        Returns:
            Tuple of (exponent, confidence/R², convergence_achieved)
        """
        # Find valid (non-zero) portion of divergence curve
        valid_mask = (divergence != 0) & np.isfinite(divergence)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 3:
            logger.warning("Insufficient valid divergence points")
            return 0.0, 0.0, False
        
        # Use first portion where divergence is approximately linear
        # (before saturation effects)
        x = valid_indices[:len(valid_indices) // 2]
        y = divergence[x]
        
        if len(x) < 3:
            return 0.0, 0.0, False
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Check convergence
        convergence = (r_value ** 2 > 0.8) and (std_err < abs(slope) * 0.5)
        
        return slope, r_value ** 2, convergence
    
    def compute_from_traces_comparison(
        self,
        trace1: np.ndarray,
        trace2: np.ndarray
    ) -> float:
        """
        Compute sensitivity by comparing two traces directly.
        
        This is a simpler method that measures how much two traces
        (from slightly perturbed inputs) diverge.
        
        Args:
            trace1: First execution trace
            trace2: Second execution trace (from perturbed input)
            
        Returns:
            Divergence rate (similar to Lyapunov exponent)
        """
        # Align trace lengths
        min_len = min(len(trace1), len(trace2))
        t1 = trace1[:min_len]
        t2 = trace2[:min_len]
        
        # Compute element-wise divergence
        diff = np.abs(t1 - t2)
        
        # Avoid log(0)
        diff = np.maximum(diff, 1e-10)
        
        # Compute log divergence
        log_diff = np.log(diff)
        
        # Estimate growth rate
        time_indices = np.arange(len(log_diff))
        valid = np.isfinite(log_diff)
        
        if np.sum(valid) < 3:
            return 0.0
        
        slope, _, _, _, _ = linregress(time_indices[valid], log_diff[valid])
        
        return slope
    
    def estimate_optimal_embedding(
        self,
        time_series: np.ndarray,
        max_dimension: int = 10
    ) -> Tuple[int, int]:
        """
        Estimate optimal embedding dimension and time delay.
        
        Uses false nearest neighbors (FNN) for dimension and
        first minimum of autocorrelation for delay.
        
        Args:
            time_series: Input time series
            max_dimension: Maximum dimension to test
            
        Returns:
            Tuple of (optimal_dimension, optimal_delay)
        """
        # Estimate time delay using first minimum of autocorrelation
        autocorr = np.correlate(time_series - np.mean(time_series), 
                                time_series - np.mean(time_series), 
                                mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find first local minimum
        optimal_delay = 1
        for i in range(1, min(len(autocorr) - 1, 50)):
            if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
                optimal_delay = i
                break
        
        # Estimate dimension using false nearest neighbors
        optimal_dim = self._estimate_dimension_fnn(time_series, optimal_delay, max_dimension)
        
        return optimal_dim, optimal_delay
    
    def _estimate_dimension_fnn(
        self,
        time_series: np.ndarray,
        delay: int,
        max_dim: int
    ) -> int:
        """Estimate embedding dimension using false nearest neighbors."""
        fnn_ratios = []
        
        for dim in range(1, max_dim + 1):
            # This is a simplified FNN implementation
            # Full implementation would require more sophisticated analysis
            try:
                self.embedding_dimension = dim
                self.time_delay = delay
                embedded = self._embed(time_series)
                
                if len(embedded) < 10:
                    break
                
                tree = KDTree(embedded)
                distances, indices = tree.query(embedded, k=2)
                
                # Count false nearest neighbors
                false_nn = 0
                total = 0
                
                for i in range(len(embedded) - delay):
                    if distances[i, 1] > 1e-10:
                        # Check if neighbor remains close in higher dimension
                        d_current = distances[i, 1]
                        
                        if dim < max_dim:
                            # Distance in next dimension
                            if i + dim * delay < len(time_series):
                                d_extra = abs(time_series[i + dim * delay] - 
                                            time_series[indices[i, 1] + dim * delay])
                                
                                # Criterion for false nearest neighbor
                                if d_extra / d_current > 10:
                                    false_nn += 1
                        total += 1
                
                fnn_ratio = false_nn / max(total, 1)
                fnn_ratios.append(fnn_ratio)
                
                # Stop if FNN ratio is low enough
                if fnn_ratio < 0.05:
                    return dim
                    
            except Exception:
                break
        
        # Return dimension with lowest FNN ratio
        if fnn_ratios:
            return np.argmin(fnn_ratios) + 1
        return 3  # Default


def compute_lyapunov_exponent(
    traces: Union[List[np.ndarray], np.ndarray],
    embedding_dimension: int = 5,
    time_delay: int = 1
) -> float:
    """
    Convenience function to compute Lyapunov exponent.
    
    Args:
        traces: Execution traces
        embedding_dimension: Phase space dimension
        time_delay: Time delay for embedding
        
    Returns:
        Largest Lyapunov exponent
    """
    lyapunov = DiscreteLyapunov(
        embedding_dimension=embedding_dimension,
        time_delay=time_delay
    )
    return lyapunov.compute(traces)
