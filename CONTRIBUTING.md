# VoiceFlow Intelligence Platform - Contributing Guide

Thank you for your interest in contributing to VoiceFlow Intelligence Platform! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## 📜 Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Rust 1.75+
- Docker & Docker Compose
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/FCHEHIDI/VoiceFlow-Intelligence-Platform.git
cd VoiceFlow-Intelligence-Platform

# Python setup
cd voiceflow-ml
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Rust setup
cd ../voiceflow-inference
cargo build

# Start services
cd ..
docker-compose up -d
```

## 🔄 Development Workflow

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Commit your changes** with clear, descriptive messages
5. **Push to your fork** and submit a pull request

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions or modifications

## ✅ Code Quality Standards

### Python

**Formatting:**
```bash
black .
isort .
```

**Linting:**
```bash
flake8 .
mypy . --ignore-missing-imports
```

**Testing:**
```bash
pytest tests/ --cov=. --cov-report=html
```

**Coverage Target:** Minimum 80%

### Rust

**Formatting:**
```bash
cargo fmt
```

**Linting:**
```bash
cargo clippy -- -D warnings
```

**Testing:**
```bash
cargo test --verbose
```

### Pre-commit Checks

We recommend using pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## 📝 Submitting Changes

### Pull Request Process

1. **Update documentation** if you're changing functionality
2. **Add tests** for new features
3. **Ensure all tests pass** locally
4. **Update the README.md** if needed
5. **Follow the PR template** when submitting

### PR Title Format

```
[Type] Brief description

Types: feat, fix, docs, style, refactor, test, chore
```

**Examples:**
- `[feat] Add A/B testing for models`
- `[fix] Resolve WebSocket connection timeout`
- `[docs] Update API documentation`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where necessary
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] All tests pass locally
```

## 🐛 Reporting Bugs

### Before Submitting

1. **Check existing issues** to avoid duplicates
2. **Test against the latest version**
3. **Collect relevant information** (logs, screenshots, etc.)

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or logs.

**Environment:**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.11]
- Rust version: [e.g., 1.75]
- Docker version: [e.g., 20.10]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

We love new ideas! Submit feature requests as GitHub issues with:

- **Clear title** and description
- **Use case** explaining why this feature would be useful
- **Proposed solution** if you have one
- **Alternatives considered**

## 🔍 Code Review Process

- All submissions require review
- Reviewers will check for:
  - Code quality and style
  - Test coverage
  - Documentation
  - Performance implications
- Address feedback promptly
- Maintain a constructive dialogue

## 📚 Additional Resources

- [README.md](README.md) - Project overview
- [CAHIER_DES_CHARGES.md](docs/CAHIER_DES_CHARGES.md) - Requirements
- [CONCEPTION_TECHNIQUE.md](docs/CONCEPTION_TECHNIQUE.md) - Technical architecture

## 🙏 Recognition

Contributors will be recognized in our [README.md](README.md) and release notes.

---

**Questions?** Open an issue or reach out to the maintainers.

Thank you for contributing! 🎤🔥
