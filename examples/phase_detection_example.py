#!/usr/bin/env python3
"""
Phase Transition Detection Example

This example demonstrates how to detect phase transitions in compiler
optimization behavior. Phase transitions indicate critical thresholds
where the compiler switches between different optimization strategies.

These transitions are high-value testing targets because bugs often
cluster at decision boundaries.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sensitivity_testing.algorithms import PhaseTransitionDetector, detect_phase_transitions


def example_1_basic_detection():
    """Basic phase transition detection using simulated data."""
    print("\n" + "="*60)
    print("Example 1: Basic Phase Transition Detection")
    print("="*60)
    
    # Create detector
    detector = PhaseTransitionDetector(
        method='cusum',
        significance_level=0.05
    )
    
    # Define a program template with variable loop bound
    # The compiler's unrolling behavior will change at certain thresholds
    template = '''
    #include <stdio.h>
    int main() {
        int sum = 0;
        int arr[{N}];
        for (int i = 0; i < {N}; i++) {
            arr[i] = i * 2;
            sum += arr[i];
        }
        printf("%d\\n", sum);
        return 0;
    }
    '''
    
    # Detect transitions across loop bounds 1-200
    print("\nScanning loop bounds 1-200 for phase transitions...")
    transitions = detector.detect(
        parameter_name='loop_trip_count',
        parameter_range=range(1, 200),
        program_template=template
    )
    
    print(f"\nDetected {len(transitions)} phase transitions:")
    for t in transitions:
        print(f"  - At N={t.parameter_value}: {t.change_type.value} (confidence: {t.confidence:.2f})")


def example_2_multiple_methods():
    """Compare different detection methods."""
    print("\n" + "="*60)
    print("Example 2: Comparing Detection Methods")
    print("="*60)
    
    import numpy as np
    
    # Simulate compiler behavior with clear phase transitions
    def simulate_behavior(n):
        """Simulate compilation time with phase transitions at n=8, 32, 64."""
        base = 1.0
        
        # Unrolling kicks in at 8
        if n >= 8:
            base *= 0.8
        
        # Vectorization at 32
        if n >= 32:
            base *= 0.6
        
        # Different strategy at 64
        if n >= 64:
            base *= 1.2  # Overhead from complex optimization
        
        # Add noise
        return base + np.random.normal(0, 0.05)
    
    methods = ['cusum', 'pelt', 'bocpd']
    
    for method in methods:
        print(f"\n--- Method: {method.upper()} ---")
        
        try:
            transitions = detect_phase_transitions(
                parameter_range=range(1, 100),
                behavior_function=simulate_behavior,
                method=method
            )
            
            print(f"Detected transitions at: {transitions}")
            print(f"Expected: ~8, ~32, ~64")
        except Exception as e:
            print(f"Error: {e}")


def example_3_real_compiler():
    """Detect phase transitions in a real compiler (requires GCC/Clang)."""
    print("\n" + "="*60)
    print("Example 3: Real Compiler Phase Detection")
    print("="*60)
    
    import shutil
    
    # Check if compiler is available
    if not shutil.which('gcc'):
        print("GCC not found. Skipping real compiler example.")
        return
    
    print("\nNote: This example requires GCC to be installed.")
    print("It will compile programs with varying array sizes and measure")
    print("the compilation time to detect optimization phase transitions.")
    
    # This would involve actual compilation - simplified here
    print("\n[This example requires actual compilation infrastructure]")
    print("[See the full framework for real compiler integration]")


def example_4_visualization():
    """Visualize phase transitions (requires matplotlib)."""
    print("\n" + "="*60)
    print("Example 4: Visualizing Phase Transitions")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Generate behavior data with transitions
        x = np.arange(1, 150)
        y = []
        
        for n in x:
            base = 10 + n * 0.1
            if n >= 16:
                base -= 3
            if n >= 64:
                base -= 5
            if n >= 100:
                base += 8
            y.append(base + np.random.normal(0, 0.3))
        
        y = np.array(y)
        
        # Detect transitions
        transitions = detect_phase_transitions(
            parameter_range=range(1, 150),
            behavior_function=lambda i: y[i-1],
            method='cusum'
        )
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, 'b-', alpha=0.7, label='Behavior metric')
        
        for t in transitions:
            plt.axvline(x=t, color='r', linestyle='--', alpha=0.7)
        
        plt.xlabel('Parameter Value (N)')
        plt.ylabel('Behavior Metric')
        plt.title('Phase Transition Detection in Compiler Behavior')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = Path(__file__).parent / 'phase_transitions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# PHASE TRANSITION DETECTION EXAMPLES")
    print("#"*60)
    
    example_1_basic_detection()
    example_2_multiple_methods()
    example_3_real_compiler()
    example_4_visualization()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
