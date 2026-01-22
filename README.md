<!-- markdownlint-disable MD033 -->

<h1 align="center">Deep Learning</h1>

<p align="center">
  <i>Technical Foundations and Modern Architectures in Neural Networks.</i><br><br>

<!-- CI Badges -->

<a href="https://github.com/kokou-egbewatt/deep-learning/actions/workflows/ci.yml">
  <img src="https://github.com/kokou-egbewatt/deep-learning/actions/workflows/ci.yml/badge.svg" alt="CI">
</a>

<!-- License Badge -->

<a href="https://github.com/kokou-egbewatt/deep-learning/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT">
</a>
</p>

## Overview

This repository is dedicated to the theoretical and practical aspects of neural networks, with a focus on both foundational models and extensible architectures. It provides:

- **Mathematical and algorithmic implementations** of core neural network concepts, starting with the perceptron as the fundamental building block for binary classification and linear separability.
- **Hands-on code** for constructing, training, and evaluating neural networks from scratch, emphasizing transparency and educational value.
- **Extensible design** to facilitate experimentation with learning rules, activation functions, and network topologies, supporting both research and teaching use cases.
- **Modern Python packaging and testing** practices, ensuring reproducibility and ease of integration into larger machine learning workflows.

The project is structured to help users understand the step-by-step mechanics of neural computation, weight updates, convergence, and the transition from single-layer to multi-layer architectures.

## Project Structure

```bash
deep-learning/
├── perceptron/
│   ├── __init__.py
│   ├── perceptron.py
│   ├── readme.md
│   └── single-layer.ipynb
├── tests/
│   └── perceptron/
│       └── test_perceptron.py
├── README.md
├── pyproject.toml
└── ...
```

## Installation

This project uses [PEP 621](https://peps.python.org/pep-0621/) and [Hatchling](https://hatch.pypa.io/) for packaging. To install dependencies:

```bash
uv sync
```

## Usage

You can use the perceptron implementation directly in your Python code:

```python
from perceptron.perceptron import Perceptron

# Example: AND logic gate
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
p = Perceptron(2, learning_rate=0.1)
p.train(X, y, epochs=20)
predictions = [p.predict(x) for x in X]
print(predictions)  # Output: [0, 0, 0, 1]
```

## Running Tests

To run the perceptron unit tests:

```bash
uv run tests tests/perceptron
```
