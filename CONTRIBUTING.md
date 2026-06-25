# Contributing to DRIMAPS

Thank you for your interest in contributing to DRIMAPS! We welcome contributions from researchers, developers, and enthusiasts.

## Ways to Contribute

- **Report bugs** — Open issues for bugs you find
- **Suggest enhancements** — Share ideas for new features or improvements
- **Submit code** — Send pull requests for bug fixes or new features
- **Improve documentation** — Help us write better docs and tutorials
- **Share results** — Report experimental results or optimizations

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/drimaps.git
cd drimaps
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/issue-description
```

## Development Guidelines

### Code Style

- Follow PEP 8 standards
- Use `black` for formatting: `black src/ sim/`
- Use `isort` for import sorting: `isort src/ sim/`
- Check with `flake8`: `flake8 src/ sim/`

### Testing

All contributions should include tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=sim

# Run specific test file
pytest tests/test_drimaps.py -v
```

### Writing Tests

- Tests should be in the `tests/` directory
- Use descriptive test names: `test_cycle_detection_simple_cycle`
- Include docstrings explaining what is being tested
- Test edge cases and error conditions

### Documentation

- Add docstrings to all functions and classes
- Update README if adding major features
- Update inline comments for complex logic
- Run `pytest --cov` to ensure good code coverage

## Commit Guidelines

- Write clear, concise commit messages
- Reference issues: `Fixes #123` or `Related to #456`
- Example: `Add cycle detection optimization for large graphs`

```
Add cycle detection optimization for large graphs

- Implement windowed WFG maintenance (issue #123)
- Reduces memory usage by 40% for 1000+ agents
- Adds comprehensive unit tests
- Updates performance documentation
```

## Pull Request Process

1. **Ensure all tests pass**: `pytest tests/ -v`
2. **Check code style**: `black --check` and `flake8`
3. **Update documentation** if needed
4. **Write a clear PR description**:
   - What problem does this solve?
   - How is it solved?
   - Any breaking changes?
   - Related issues or PRs

5. **Request review** from maintainers
6. **Address feedback** promptly

## Reporting Issues

When reporting bugs, please include:

- **Description**: Clear explanation of the issue
- **Steps to reproduce**: Minimal reproducible example
- **Environment**: Python version, OS, hardware (agents/map size)
- **Expected vs. actual behavior**
- **Relevant logs or error messages**
- **Screenshots** (if applicable)

### Example Bug Report

```
Title: Deadlock detection fails with >256 agents

Description:
Cycle detection reports false deadlocks when solving instances
with more than 256 agents on warehouse maps.

Steps to Reproduce:
```python
env = drimapsim_v0(EnvConfig(
    size=64, num_agents=256, map_type="warehouse"
))
# After ~100 steps, crashes with "invalid agent index"
```

Environment:
- Python 3.10
- DRIMAPS 1.0.0
- Ubuntu 22.04 LTS
- 16GB RAM, 8 CPU cores

Error:
```
IndexError: index 256 out of range for array of size 256
  in cycle_detector.py:145
```
```

## Feature Requests

For feature suggestions:

- Check existing issues first
- Explain the use case
- Discuss potential implementation approach
- Be open to feedback and alternatives

## Research Contributions

Publishing new experiments or findings?

1. **Run official experiments**: `python experiments/run_benchmark.py` (then `python analysis/make_paper_assets.py`)
2. **Document methodology**: Note any new instance/map settings in `experiments/`
3. **Share results**: Create new analysis in `analysis/`
4. **Submit paper appendix** or supplementary materials

## Questions?

- Check the [Wiki](https://github.com/your-username/drimaps/wiki)
- Open a Discussion for questions
- Email: mapf@research.org

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Code of Conduct

Be respectful and inclusive. We're building a community for all researchers and developers interested in Multi-Agent Path Finding.

---

**Thank you for contributing to DRIMAPS!** 🎉
