# Contributing to Sensitivity-Theoretic Compiler Testing

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sensitivity-compiler-testing.git
   cd sensitivity-compiler-testing
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- **Python Style**: Follow PEP 8 guidelines
- **Formatting**: Use `black` for code formatting
- **Type Hints**: Add type hints to all function signatures
- **Docstrings**: Use Google-style docstrings

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Follow conventional commit format:

```
type(scope): subject

body (optional)

footer (optional)
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(lyapunov): add optimal embedding estimation

Implements false nearest neighbors method for automatic
selection of embedding dimension.

Closes #42
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** and add tests

3. **Run the test suite**
   ```bash
   pytest tests/ -v --cov=sensitivity_testing
   ```

4. **Format and lint**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

6. **Fill out the PR template** with:
   - Description of changes
   - Related issues
   - Test coverage

## Testing Guidelines

- Write tests for all new functionality
- Maintain >80% code coverage
- Include unit tests and integration tests where appropriate
- Use descriptive test names: `test_<function>_<scenario>_<expected>`

```python
def test_lyapunov_chaotic_system_returns_positive_exponent():
    """Test that known chaotic systems produce positive Lyapunov exponents."""
    ...
```

## Documentation

- Update docstrings for all public APIs
- Add examples for new features
- Update README.md if adding major features

## Questions?

Open an issue for questions or discussions about contributions.

Thank you for contributing!
