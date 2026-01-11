"""
Sensitivity-Theoretic Compiler Testing Framework

A novel framework for compiler testing that applies chaos theory and 
dynamical systems analysis to identify bug-prone regions in compiler 
optimization pipelines.

Key Components:
    - DiscreteLyapunov: Computes sensitivity exponents from execution traces
    - PhaseTransitionDetector: Detects critical parameter boundaries
    - SensitivityOracle: Prioritizes test inputs with PAC bounds
    - SensitivityFramework: Main orchestration class

Example:
    >>> from sensitivity_testing import SensitivityFramework
    >>> framework = SensitivityFramework(compilers=['gcc', 'clang'])
    >>> result = framework.analyze('test.c')
    >>> print(result.lyapunov_exponent)
"""

__version__ = "0.1.0"
__author__ = "H. M. Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from .framework import SensitivityFramework
from .algorithms.lyapunov import DiscreteLyapunov
from .algorithms.phase_transition import PhaseTransitionDetector
from .algorithms.sensitivity_oracle import SensitivityOracle

__all__ = [
    "SensitivityFramework",
    "DiscreteLyapunov",
    "PhaseTransitionDetector", 
    "SensitivityOracle",
    "__version__",
]
