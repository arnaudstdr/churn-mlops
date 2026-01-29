# Contributing to Churn MLOps

Thank you for your interest in contributing to the Churn MLOps project! This document outlines the guidelines and conventions to follow when contributing to this project.

## Table of Contents

- [Contributing to Churn MLOps](#contributing-to-churn-mlops)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Submitting Pull Requests](#submitting-pull-requests)
  - [Development Setup](#development-setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Coding Conventions](#coding-conventions)
    - [General Guidelines](#general-guidelines)
    - [Naming Conventions](#naming-conventions)
    - [Code Style](#code-style)
    - [Example](#example)
  - [Commit Guidelines](#commit-guidelines)
  - [Branch Naming](#branch-naming)
  - [Pull Request Process](#pull-request-process)
  - [Versioning](#versioning)
  - [Documentation](#documentation)
    - [Updating Documentation](#updating-documentation)
    - [Docstrings](#docstrings)
  - [Testing](#testing)
    - [Running Tests](#running-tests)
    - [Writing Tests](#writing-tests)
  - [Questions or Issues](#questions-or-issues)

## Code of Conduct

By participating in this project, you agree to abide by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). Please read it to understand the expected behavior.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with the following information:

- A clear and descriptive title.
- Steps to reproduce the bug.
- Expected behavior.
- Actual behavior.
- Any relevant screenshots or logs.

### Suggesting Enhancements

If you have an idea for a new feature or enhancement, please open an issue on GitHub with the following information:

- A clear and descriptive title.
- A detailed description of the proposed feature.
- The rationale behind the feature.
- Any relevant examples or mockups.

### Submitting Pull Requests

Pull requests are welcome! Please follow the [Pull Request Process](#pull-request-process) below.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for running MLflow)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arnaudstadler/churn-mlops.git
   cd churn-mlops
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Coding Conventions

### General Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
- Use type hints for functions and methods.
- Write clear and descriptive docstrings for all public functions, classes, and modules.
- Keep lines under 88 characters.

### Naming Conventions

- **Variables and Functions**: Use `snake_case`.
  - Example: `customer_id`, `calculate_churn_probability()`
- **Classes**: Use `PascalCase`.
  - Example: `CustomerChurnModel`, `APIConfig`
- **Files and Directories**: Use lowercase with underscores for files and hyphens for directories.
  - Example: `api/schemas.py`, `data/raw/`

### Code Style

- Use 4 spaces for indentation.
- Group imports by type (standard library, third-party, local) with a blank line between each group.
- Avoid using wildcard imports (`from module import *`).

### Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class CustomerChurnModel:
    """A model for predicting customer churn."""

    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model.

        Args:
            X: Features for training.
            y: Target variable.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict churn probabilities.

        Args:
            X: Features for prediction.

        Returns:
            Predicted probabilities.
        """
        return self.model.predict_proba(X)[:, 1]
```

## Commit Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. The format is:

```
type(scope optional): description

Examples:
- feat(api): add POST /predict endpoint
- fix(ml): correct preprocessing bug
- docs: update README
- refactor(db): simplify queries

Allowed types:
- feat: A new feature
- fix: A bug fix
- docs: Documentation changes
- style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc.)
- refactor: A code change that neither fixes a bug nor adds a feature
- perf: A code change that improves performance
- test: Adding missing tests or correcting existing tests
- build: Changes that affect the build system or external dependencies
- ci: Changes to CI configuration files and scripts
- chore: Other changes that don't modify src or test files
- revert: Reverts a previous commit
```

## Branch Naming

Use the following format for branch names:

```
type/short-description

Examples:
- feat/api-add-predict-endpoint
- fix/ml-correct-preprocessing
- docs/update-readme
```

## Pull Request Process

1. Fork the repository and create a new branch from `main`.
2. Make your changes and ensure they follow the coding conventions.
3. Write tests for your changes if applicable.
4. Update the documentation if necessary.
5. Commit your changes with a clear and descriptive message following the commit guidelines.
6. Push your branch to your fork.
7. Open a pull request to the `main` branch of the original repository.
8. Wait for the review and address any feedback.

## Versioning

This project uses [Semantic Versioning](https://semver.org/) (SemVer) for versioning. The format is `vX.Y.Z` where:

- **X**: Major version (incompatible changes)
- **Y**: Minor version (backwards-compatible features)
- **Z**: Patch version (backwards-compatible bug fixes)

To bump the version, use `bumpversion`:

```bash
# Bump patch version (e.g., 0.1.0 → 0.1.1)
bumpversion patch

# Bump minor version (e.g., 0.1.0 → 0.2.0)
bumpversion minor

# Bump major version (e.g., 0.1.0 → 1.0.0)
bumpversion major
```

## Documentation

### Updating Documentation

- Keep the `README.md` up to date with the latest information.
- Update the `CHANGELOG.md` with notable changes for each release.
- Add or update documentation in the `docs/` directory as needed.

### Docstrings

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public functions, classes, and modules.

Example:

```python
def calculate_churn_probability(customer_data: dict) -> float:
    """Calculate the probability of customer churn.

    Args:
        customer_data: A dictionary containing customer features.

    Returns:
        The predicted churn probability.

    Raises:
        ValueError: If required features are missing.
    """
    ...
```

## Testing

### Running Tests

To run the tests, use the following command:

```bash
pytest
```

### Writing Tests

- Write tests for new features and bug fixes.
- Use descriptive test names.
- Group related tests in test classes.

Example:

```python
import pytest
from api.service import predict_churn

def test_predict_churn():
    """Test the predict_churn function."""
    customer_data = {"customer_id": "123", "features": {"tenure": 12, "monthly_charges": 50}}
    result = predict_churn(customer_data)
    assert isinstance(result, float)
    assert 0 <= result <= 1
```

## Questions or Issues

If you have any questions or run into issues, please open an issue on GitHub or contact the maintainers.

Thank you for contributing to Churn MLOps! 🚀