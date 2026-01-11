#!/usr/bin/env python3
"""
Run an experiment with the sensitivity-theoretic compiler testing framework.

Usage:
    python run_experiment.py --config experiments/configs/default_experiment.yaml
    python run_experiment.py --compilers gcc,clang --budget 24 --generator csmith
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sensitivity_testing import SensitivityFramework
from sensitivity_testing.oracles import DifferentialOracle, CrashOracle
from sensitivity_testing.core.program import ProgramGenerator
from sensitivity_testing.utils.config import FrameworkConfig
from sensitivity_testing.utils.logging import setup_logging


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_oracle(oracle_config: dict):
    """Create oracle based on configuration."""
    oracle_type = oracle_config.get('type', 'differential')
    
    if oracle_type == 'differential':
        return DifferentialOracle()
    elif oracle_type == 'crash':
        return CrashOracle(check_sanitizers=oracle_config.get('check_sanitizers', True))
    else:
        return DifferentialOracle()


def main():
    parser = argparse.ArgumentParser(description='Run sensitivity-theoretic compiler testing experiment')
    parser.add_argument('--config', type=str, help='Path to experiment config YAML')
    parser.add_argument('--compilers', type=str, default='gcc,clang', help='Comma-separated compiler list')
    parser.add_argument('--budget', type=float, default=1.0, help='Budget in hours')
    parser.add_argument('--generator', type=str, default='simple', help='Program generator type')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    # Load config or use command line args
    if args.config:
        config = load_config(args.config)
        compilers = config.get('compilers', ['gcc', 'clang'])
        budget_hours = config.get('budget', {}).get('hours', 1.0)
        generator_type = config.get('generator', {}).get('type', 'simple')
        oracle = create_oracle(config.get('oracle', {}))
        output_dir = Path(config.get('output', {}).get('results_dir', './results'))
    else:
        compilers = args.compilers.split(',')
        budget_hours = args.budget
        generator_type = args.generator
        oracle = DifferentialOracle()
        output_dir = Path(args.output)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Sensitivity-Theoretic Compiler Testing Framework")
    print(f"{'='*60}")
    print(f"Compilers: {compilers}")
    print(f"Budget: {budget_hours} hours")
    print(f"Generator: {generator_type}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")
    
    # Initialize framework
    framework = SensitivityFramework(
        compilers=compilers,
        optimization_levels=['-O0', '-O1', '-O2', '-O3']
    )
    
    # Create generator
    generator = ProgramGenerator.create(generator_type)
    
    # Run testing campaign
    print("Starting testing campaign...")
    result = framework.run_prioritized_testing(
        test_generator=generator,
        budget_hours=budget_hours,
        oracle=oracle,
        output_dir=run_dir / 'bugs'
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Programs tested: {result.programs_tested}")
    print(f"Bugs found: {len(result.bugs_found)}")
    print(f"Bugs per hour: {result.bugs_per_hour:.2f}")
    print(f"Coverage achieved: {result.coverage_achieved:.2%}")
    print(f"Phase transitions: {len(result.phase_transitions_discovered)}")
    print(f"{'='*60}\n")
    
    # Save results
    import json
    with open(run_dir / 'results.json', 'w') as f:
        json.dump({
            'programs_tested': result.programs_tested,
            'bugs_found': len(result.bugs_found),
            'bugs_per_hour': result.bugs_per_hour,
            'coverage_achieved': result.coverage_achieved,
            'phase_transitions': len(result.phase_transitions_discovered),
            'bugs': result.bugs_found
        }, f, indent=2)
    
    print(f"Results saved to: {run_dir}")


if __name__ == '__main__':
    main()
