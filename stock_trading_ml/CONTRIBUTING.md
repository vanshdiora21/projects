# Contributing to ML Trading Strategy

We welcome contributions to the ML Trading Strategy project! This document provides guidelines for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:

git clonehttps://github.com/vanshdiora21/projects/tree/main/stock_trading_ml
cd ml-trading-strategy

text

3. Create a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

text

4. Install dependencies:
pip install -r requirements.txt
pip install -e . # Install in development mode

text

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Keep functions focused and modular

## Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting PR:
python -m pytest tests/ -v

text
- Aim for high test coverage

## Submitting Changes

1. Create a feature branch:
git checkout -b feature/your-feature-name

text

2. Make your changes and commit:
git add .
git commit -m "Add your descriptive commit message"

text

3. Push to your fork:
git push origin feature/your-feature-name

text

4. Create a Pull Request on GitHub

## Reporting Issues

- Use the GitHub issue tracker
- Provide clear, detailed descriptions
- Include system information and error messages
- Add steps to reproduce the issue

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors

Thank you for contributing!