#!/usr/bin/env python3
"""
Basic usage example for the Sensitivity-Theoretic Compiler Testing Framework.

This example demonstrates:
1. Analyzing a single program for sensitivity
2. Interpreting the results
3. Basic phase transition detection

Requirements:
    - Python 3.9+
    - GCC and/or Clang installed
    - pip install sensitivity-compiler-testing
"""

import tempfile
from pathlib import Path

# Import the framework
from sensitivity_testing import SensitivityFramework, DiscreteLyapunov
from sensitivity_testing.core.program import TestProgram


def main():
    """Main example function."""
    
    print("=" * 60)
    print("Sensitivity-Theoretic Compiler Testing - Basic Example")
    print("=" * 60)
    
    # Example 1: Create a simple test program
    print("\n1. Creating test program...")
    
    test_code = '''
#include <stdio.h>

int main() {
    int sum = 0;
    int arr[100];
    
    // Initialize array
    for (int i = 0; i < 100; i++) {
        arr[i] = i * 3 + 7;
    }
    
    // Compute sum with some operations
    for (int i = 0; i < 100; i++) {
        sum += arr[i];
        if (arr[i] % 2 == 0) {
            sum *= 2;
            sum /= 2;
        }
    }
    
    printf("Result: %d\\n", sum);
    return 0;
}
'''
    
    # Create test program from source
    program = TestProgram.from_source(test_code, name="basic_test")
    print(f"   Created: {program.name}")
    print(f"   Lines: {program.line_count()}")
    print(f"   Parameters: {program.extract_parameters()}")
    
    # Example 2: Initialize the framework
    print("\n2. Initializing framework...")
    
    try:
        framework = SensitivityFramework(
            compilers=['gcc', 'clang'],
            optimization_levels=['-O0', '-O2', '-O3']
        )
        print(f"   Available compilers: {framework.compiler_manager.available_compilers}")
    except RuntimeError as e:
        print(f"   Warning: {e}")
        print("   Continuing with demonstration...")
        framework = None
    
    # Example 3: Direct Lyapunov computation (without compilation)
    print("\n3. Computing Lyapunov exponent from simulated traces...")
    
    import numpy as np
    
    # Simulate some execution traces (in real usage, these come from actual compilation)
    traces = [
        np.random.randn(100) + np.sin(np.linspace(0, 10, 100)),  # Trace 1
        np.random.randn(100) + np.sin(np.linspace(0, 10, 100)) * 1.1,  # Slightly different
        np.random.randn(100) + np.sin(np.linspace(0, 10, 100)) * 0.9,  # Slightly different
    ]
    
    lyapunov = DiscreteLyapunov(embedding_dimension=3, time_delay=1)
    result = lyapunov.compute(traces, return_diagnostics=True)
    
    print(f"   Lyapunov exponent (λ): {result.exponent:.4f}")
    print(f"   Confidence (R²): {result.confidence:.4f}")
    print(f"   Interpretation: {result.interpretation()}")
    
    if result.is_chaotic():
        print("   ⚠️  CHAOTIC: This region is potentially bug-prone!")
    else:
        print("   ✓  STABLE: This region appears robust")
    
    # Example 4: Phase transition detection
    print("\n4. Detecting phase transitions...")
    
    from sensitivity_testing.algorithms import PhaseTransitionDetector
    
    detector = PhaseTransitionDetector(method='cusum')
    
    # Simulate behavior across parameter range
    # (In real usage, this would compile programs with varying parameters)
    def simulate_compiler_behavior(param_value):
        """Simulate compiler behavior metric."""
        base = 1.0
        
        # Simulate loop unrolling threshold at 8
        if param_value >= 8:
            base *= 0.85
        
        # Simulate vectorization threshold at 64
        if param_value >= 64:
            base *= 0.6
        
        # Add noise
        return base + np.random.normal(0, 0.02)
    
    parameter_range = range(1, 100)
    metrics = [simulate_compiler_behavior(v) for v in parameter_range]
    
    change_points = detector._detect_cusum(np.array(metrics))
    
    print(f"   Detected {len(change_points)} potential phase transitions")
    for cp in change_points:
        if cp < len(parameter_range):
            print(f"   - At parameter value ~{list(parameter_range)[cp]}")
    
    # Example 5: PAC learning bounds
    print("\n5. Computing PAC learning bounds...")
    
    from sensitivity_testing.algorithms import SensitivityOracle
    
    oracle = SensitivityOracle(epsilon=0.05, delta=0.01)
    
    # Example sensitivity scores
    sensitivity_scores = [0.1, 0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6, 0.9, 0.15]
    
    required_tests = oracle.required_tests(sensitivity_scores, target_coverage=0.95)
    coverage_bound = oracle.compute_coverage_bound(num_tests=100)
    
    print(f"   For 95% coverage confidence, need ~{required_tests} tests")
    print(f"   With 100 tests: coverage ≥ {coverage_bound.lower_bound:.2%} (confidence: {coverage_bound.confidence:.2%})")
    
    # Example 6: Bug probability estimation
    print("\n6. Estimating bug probabilities...")
    
    high_sensitivity = 0.8
    low_sensitivity = -0.3
    
    p_bug_high = oracle.estimate_bug_probability(high_sensitivity)
    p_bug_low = oracle.estimate_bug_probability(low_sensitivity)
    
    print(f"   High sensitivity (λ={high_sensitivity}): P(bug) = {p_bug_high:.2%}")
    print(f"   Low sensitivity (λ={low_sensitivity}): P(bug) = {p_bug_low:.2%}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    
    # Cleanup
    if program.path.exists():
        program.path.unlink()


if __name__ == '__main__':
    main()
