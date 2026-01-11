"""
Core algorithms for sensitivity-theoretic compiler testing.

This package provides three main algorithms:

1. DiscreteLyapunov: Computes sensitivity exponents from execution traces
   Complexity: O(T log T) where T is trace length
   
2. PhaseTransitionDetector: Detects critical parameter boundaries
   Complexity: O(n) where n is parameter range size
   
3. SensitivityOracle: Prioritizes tests with PAC learning bounds
   Provides theoretical guarantees on bug-finding probability
"""

from .lyapunov import DiscreteLyapunov, LyapunovResult, compute_lyapunov_exponent
from .phase_transition import (
    PhaseTransitionDetector,
    PhaseTransition,
    ChangeType,
    detect_phase_transitions
)
from .sensitivity_oracle import (
    SensitivityOracle,
    PrioritizationResult,
    CoverageBound,
    compute_pac_bound
)

__all__ = [
    # Lyapunov
    'DiscreteLyapunov',
    'LyapunovResult',
    'compute_lyapunov_exponent',
    
    # Phase Transition
    'PhaseTransitionDetector',
    'PhaseTransition',
    'ChangeType',
    'detect_phase_transitions',
    
    # Sensitivity Oracle
    'SensitivityOracle',
    'PrioritizationResult',
    'CoverageBound',
    'compute_pac_bound',
]
