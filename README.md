# Sensitivity-Theoretic Compiler Testing Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

A novel framework for compiler testing that applies chaos theory and dynamical systems analysis to identify bug-prone regions in compiler optimization pipelines. This implementation provides algorithms for computing discrete Lyapunov exponents, detecting phase transitions, and prioritizing test inputs based on sensitivity metrics.

## 🎯 Overview

Traditional compiler testing approaches (random testing, coverage-guided fuzzing, EMI) explore input spaces without principled guidance about **where bugs concentrate**. This framework introduces sensitivity-theoretic analysis to identify high-yield testing regions.

### Key Insight

Compiler bugs cluster at **decision boundaries**—parameter values where optimization strategies qualitatively change. These boundaries exhibit chaos-theoretic properties: small input perturbations cause exponential divergence in compiler behavior. By mapping this sensitivity landscape, we concentrate testing effort on high-yield regions.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Test Program Generator                        │
│                   (Csmith, YARPGen, Custom)                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Sensitivity Analysis Engine                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │DiscreteLyapunov │  │PhaseTransition  │  │SensitivityOracle│ │
│  │  (k-d tree)     │  │     O(n)        │  │  PAC Bounds     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Prioritized Test Queue                        │
│            (Ranked by sensitivity scores + coverage)            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Bug Detection Oracles                       │
│     Differential | EMI | Metamorphic | Crash | Sanitizer        │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Features

### Core Algorithms

1. **DiscreteLyapunov** - Computes sensitivity exponents from execution traces (O(T log T) via k-d tree implementation)
2. **PhaseTransition** - Detects critical parameter boundaries in O(n) time  
3. **SensitivityOracle** - Prioritizes test inputs with PAC learning-derived coverage bounds

### Supported Compilers

- GCC (4.x - 14.x)
- LLVM/Clang (6.0 - 18.x)
- MSVC (2019, 2022)
- ICC (19.x - 2024.x)

### Bug Detection Oracles

- **Differential Testing**: Cross-compiler disagreement detection
- **EMI Oracle**: Equivalence Modulo Inputs validation
- **Metamorphic Oracle**: Optimization-level equivalence checking
- **Crash Detection**: Sanitizer integration (ASan, UBSan, MSan)
- **Output Validation**: Semantic correctness verification

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- GCC and/or Clang installed
- NumPy, SciPy, scikit-learn

### Quick Install

```bash
# Clone the repository
git clone https://github.com/hmshujaatzaheer/sensitivity-compiler-testing.git
cd sensitivity-compiler-testing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "from sensitivity_testing import DiscreteLyapunov; print('Installation successful!')"
```

## 🔧 Quick Start

### Basic Usage

```python
from sensitivity_testing import SensitivityFramework
from sensitivity_testing.oracles import DifferentialOracle

# Initialize framework
framework = SensitivityFramework(
    compilers=['gcc', 'clang'],
    optimization_levels=['-O0', '-O1', '-O2', '-O3']
)

# Analyze a test program
result = framework.analyze('test_program.c')

# Get sensitivity score
print(f"Lyapunov exponent: {result.lyapunov_exponent}")
print(f"Phase transitions detected: {result.phase_transitions}")
print(f"Bug probability estimate: {result.bug_probability}")

# Run prioritized testing
bugs = framework.run_prioritized_testing(
    test_generator='csmith',
    budget_hours=24,
    oracle=DifferentialOracle()
)
```

### Command Line Interface

```bash
# Analyze single file
sct analyze program.c --compilers gcc,clang --output results.json

# Run sensitivity-guided fuzzing campaign
sct fuzz --generator csmith --budget 24h --output bugs/

# Detect phase transitions in optimization behavior
sct phase-detect --parameter "loop-unroll-count" --range 1:100

# Generate sensitivity landscape visualization
sct visualize --input results.json --output landscape.png
```

## 📊 Algorithms

### 1. Discrete Lyapunov Exponent Computation

Adapts the Rosenstein et al. (1993) algorithm for compiler execution traces:

```python
from sensitivity_testing.algorithms import DiscreteLyapunov

# Compute Lyapunov exponent from execution traces
lyapunov = DiscreteLyapunov(
    embedding_dimension=3,
    time_delay=1,
    min_neighbors=5
)

traces = [compile_and_trace(program, opt_level) for opt_level in optimization_levels]
exponent = lyapunov.compute(traces)

# Interpretation:
# λ > 0: Chaotic behavior (bug-prone region)
# λ < 0: Stable behavior (likely correct)
# λ ≈ 0: Edge of chaos (phase transition boundary)
```

**Complexity**: O(T log T) where T is trace length (achieved via k-d tree nearest-neighbor search in our implementation)

### 2. Phase Transition Detection

Identifies critical parameter boundaries using change-point analysis:

```python
from sensitivity_testing.algorithms import PhaseTransitionDetector

detector = PhaseTransitionDetector(
    method='cusum',  # or 'pelt', 'bocpd'
    significance_level=0.05
)

# Scan parameter space
transitions = detector.detect(
    parameter_name='loop_trip_count',
    parameter_range=range(1, 1000),
    program_template='for(int i=0; i<{N}; i++) sum += arr[i];'
)

for t in transitions:
    print(f"Phase transition at {t.parameter_value}: {t.behavior_change}")
```

**Complexity**: O(n) where n is parameter range size

### 3. Sensitivity Oracle with PAC Bounds

Provides theoretical guarantees on bug-finding probability:

```python
from sensitivity_testing.algorithms import SensitivityOracle

oracle = SensitivityOracle(
    epsilon=0.05,  # Error tolerance
    delta=0.01,    # Confidence parameter
    hypothesis_class='vc_dimension_based'
)

# Get required test budget for desired coverage
budget = oracle.required_tests(
    sensitivity_scores=computed_scores,
    target_coverage=0.95
)

# Prioritize test queue
prioritized_queue = oracle.prioritize(
    test_candidates=generated_tests,
    sensitivity_scores=computed_scores
)
```

## 📁 Project Structure

```
sensitivity-compiler-testing/
├── src/
│   ├── sensitivity_testing/
│   │   ├── __init__.py
│   │   ├── framework.py          # Main framework orchestration
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── compiler.py       # Compiler abstraction layer
│   │   │   ├── trace.py          # Execution trace collection
│   │   │   └── program.py        # Test program representation
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   ├── lyapunov.py       # Discrete Lyapunov computation
│   │   │   ├── phase_transition.py # Phase transition detection
│   │   │   └── sensitivity_oracle.py # PAC-based prioritization
│   │   ├── oracles/
│   │   │   ├── __init__.py
│   │   │   ├── differential.py   # Cross-compiler oracle
│   │   │   ├── emi.py            # EMI oracle
│   │   │   ├── metamorphic.py    # Metamorphic testing oracle
│   │   │   └── crash.py          # Crash/sanitizer oracle
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── landscape.py      # Sensitivity landscape mapping
│   │   │   ├── clustering.py     # Bug cluster analysis
│   │   │   └── visualization.py  # Result visualization
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging.py        # Logging utilities
│   │       ├── config.py         # Configuration management
│   │       └── metrics.py        # Performance metrics
├── tests/
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── fixtures/                 # Test fixtures
├── experiments/
│   ├── benchmarks/               # Benchmark programs
│   ├── results/                  # Experimental results
│   └── configs/                  # Experiment configurations
├── docs/
│   ├── api/                      # API documentation
│   ├── guides/                   # User guides
│   └── figures/                  # Documentation figures
├── scripts/
│   ├── setup_compilers.sh        # Compiler setup script
│   ├── run_experiments.py        # Experiment runner
│   └── analyze_results.py        # Result analysis
├── examples/
│   ├── basic_usage.py            # Basic usage example
│   ├── custom_oracle.py          # Custom oracle example
│   └── phase_detection.py        # Phase detection example
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Modern Python packaging
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sensitivity_testing --cov-report=html

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
```

**Note**: 54 tests are provided covering all core algorithms. Tests pass 100%. Coverage on core algorithm modules averages 67% (SensitivityOracle: 93%, Lyapunov: 66%, PhaseTransition: 41%). Lower coverage on I/O modules is expected as those require actual compiler execution.

## 📈 Experimental Status

> **Note**: This is a research prototype implementation accompanying a PhD proposal. The framework implements the proposed algorithms but has not yet been empirically validated on production compilers.

### Planned Experiments (To Be Conducted)

The following experiments are planned as part of the PhD research:

1. **RQ1 - Correlation Validation**: Testing whether high-sensitivity regions contain more bugs
2. **RQ2 - Phase Transition Utility**: Validating bug clustering near optimization boundaries
3. **RQ3 - Comparative Efficiency**: Benchmarking against Csmith, AFL++, and coverage-guided baselines
4. **RQ4 - Complementarity**: Measuring unique bugs found by sensitivity-guided vs. coverage-guided approaches
5. **RQ5 - Scalability**: Overhead measurement on SPEC CPU2017 benchmarks

### Theoretical Basis

The framework's complexity analysis:
- **DiscreteLyapunov**: O(T log T) - achieved through k-d tree nearest-neighbor search (our implementation)
- **PhaseTransition**: O(n) - CUSUM and PELT algorithms
- **SensitivityOracle**: O(|P| log |P|) - priority queue operations

### Phase Transition Detection

The framework is designed to identify optimization thresholds such as:

- Loop unrolling boundaries (trip count thresholds)
- Inlining decisions (function size thresholds)
- Vectorization triggers (array size thresholds)
- Register allocation pressure points

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/hmshujaatzaheer/sensitivity-compiler-testing.git
cd sensitivity-compiler-testing
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests before submitting PR
pytest tests/ -v
black src/ tests/
flake8 src/ tests/
```

## 📜 Citation

If you use this framework in your research, please cite:

```bibtex
@software{sensitivity_compiler_testing,
  author = {Zaheer, H. M. Shujaat},
  title = {Sensitivity-Theoretic Compiler Testing Framework},
  year = {2025},
  url = {https://github.com/hmshujaatzaheer/sensitivity-compiler-testing}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [Csmith](https://github.com/csmith-project/csmith) - Random C program generator
- [YARPGen](https://github.com/intel/yarpgen) - Yet Another Random Program Generator
- [AFL++](https://github.com/AFLplusplus/AFLplusplus) - Coverage-guided fuzzer
- [CompCert](https://github.com/AbsInt/CompCert) - Formally verified C compiler

## 📧 Contact

- **Author**: H. M. Shujaat Zaheer
- **Email**: shujabis@gmail.com
- **GitHub**: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)

---

**Important Disclaimer**: This is a research prototype accompanying a PhD proposal to ETH Zurich. While the framework implements the proposed algorithms, it has not yet been empirically validated on production compilers. The theoretical foundations are based on Rosenstein et al. (1993) for Lyapunov exponent computation and Valiant (1984) for PAC learning bounds. Complexity claims (O(T log T) for Lyapunov computation) are based on our k-d tree implementation, not from the original Rosenstein paper which only describes the algorithm as "fast."
