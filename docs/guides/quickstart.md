# Quick Start Guide

This guide will help you get started with the Sensitivity-Theoretic Compiler Testing Framework in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- GCC and/or Clang compiler
- pip package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/hmshujaatzaheer/sensitivity-compiler-testing.git
cd sensitivity-compiler-testing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Your First Analysis

### Option 1: Python API

```python
from sensitivity_testing import SensitivityFramework

# Initialize framework
framework = SensitivityFramework(compilers=['gcc', 'clang'])

# Analyze a C program
result = framework.analyze('your_program.c')

# Check results
print(f"Lyapunov exponent: {result.lyapunov_exponent}")
print(f"Bug probability: {result.bug_probability}")

if result.is_bug_prone():
    print("⚠️ This program is in a bug-prone region!")
```

### Option 2: Command Line

```bash
# Analyze a file
sct analyze program.c --compilers gcc,clang --output results.json

# Run a 1-hour fuzzing campaign
sct fuzz --generator simple --budget 1h --output bugs/

# Detect phase transitions
sct phase-detect --parameter loop_count --range 1:100
```

## Understanding Results

### Lyapunov Exponent (λ)

| Value | Meaning |
|-------|---------|
| λ > 0.5 | Highly chaotic - likely bug-prone |
| 0 < λ < 0.5 | Moderately sensitive |
| λ ≈ 0 | Edge of stability - phase transition |
| λ < 0 | Stable - less likely to have bugs |

### Phase Transitions

Phase transitions indicate optimization thresholds:
- Loop unrolling decisions
- Function inlining thresholds  
- Vectorization triggers
- Register allocation changes

These are high-value testing targets!

## Running Experiments

```bash
# Use default configuration
python scripts/run_experiment.py --config experiments/configs/default_experiment.yaml

# Quick test (1 hour)
python scripts/run_experiment.py --budget 1 --generator simple --output results/
```

## Next Steps

1. Read the [full documentation](../api/README.md)
2. Explore [examples](../../examples/)
3. Run [benchmark experiments](../../experiments/)
4. [Contribute](../../CONTRIBUTING.md) to the project

## Getting Help

- Open an issue on GitHub
- Email: shujabis@gmail.com
