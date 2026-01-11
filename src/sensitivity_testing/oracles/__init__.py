"""
Bug detection oracles for compiler testing.

Oracles determine whether a test execution reveals a bug.
"""

from .differential import DifferentialOracle
from .emi import EMIOracle
from .metamorphic import MetamorphicOracle
from .crash import CrashOracle

__all__ = ['DifferentialOracle', 'EMIOracle', 'MetamorphicOracle', 'CrashOracle']
