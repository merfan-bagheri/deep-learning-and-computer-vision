# 🧠 HW2 Part 1: Deep Learning Foundations & Modular Framework from Scratch

Implementation of core deep learning primitives from scratch using pure NumPy and Python, including custom forward/backward propagation engines, layer abstractions, loss functions, and optimization solvers.

## 📌 Covered Concepts
- **Modular Layer Architecture**: Object-oriented base classes for neural network layers (`Linear`, `ReLU`, `Sigmoid`, `SoftmaxWithLoss`).
- **Analytical Gradient Computation**: Exact backpropagation derivations and vectorized chain-rule implementations.
- **Modular Solvers**: Training loop engine with mini-batch sampling, epoch scheduling, and loss tracking.

## 📂 Structure
- `HW2_Part1_Neural_Network_Layers.ipynb`: Interactive notebook validating gradients and multi-layer perceptron (MLP) training.
- `layers.py`: Forward and backward pass implementations of neural network layers.
- `optimizer.py`: First-order optimization routines (SGD, Momentum).
- `solver.py`: Modular training loop controller.
