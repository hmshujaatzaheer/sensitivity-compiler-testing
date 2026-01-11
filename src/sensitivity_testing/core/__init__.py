"""
Core components for compiler interaction and data representation.

This package provides:
- CompilerManager: Unified interface for multiple compilers
- TraceCollector: Execution trace collection
- TestProgram: Test program representation and generation
"""

from .compiler import CompilerManager, CompilerInfo, CompilationResult
from .trace import TraceCollector, ExecutionTrace
from .program import TestProgram, ProgramGenerator

__all__ = [
    'CompilerManager',
    'CompilerInfo', 
    'CompilationResult',
    'TraceCollector',
    'ExecutionTrace',
    'TestProgram',
    'ProgramGenerator',
]
