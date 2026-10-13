# Multilayer Perceptron (MLP) in C++

A modular, high-performance implementation of a **Multilayer Perceptron (MLP)** artificial neural network built from scratch in C++, trained using the **Online Backpropagation** algorithm with momentum and early stopping.

Developed for the *Computational Models* (*Modelos de Computación*) course at the **University of Córdoba (UCO)**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Prerequisites & Build](#prerequisites--build)
- [Usage & Command-Line Options](#usage--command-line-options)
- [Datasets](#datasets)
- [Execution Examples](#execution-examples)
- [Automation & Plotting Scripts](#automation--plotting-scripts)
- [Report & Documentation](#report--documentation)

---

## Overview

This project implements a fully configurable feed-forward Multilayer Perceptron capable of solving both classification (e.g., non-linearly separable logic gates such as XOR) and non-linear regression problems (function approximation and real-world time series).

The network is trained using stochastic/online gradient descent where weights are updated pattern by pattern, incorporating a momentum term to accelerate convergence and navigate error plateaus.

---

## Key Features

- **Flexible Architecture**: Supports networks with $H \ge 2$ layers (1 input layer, $L \ge 1$ hidden layers with uniform width, and 1 output layer).
- **Sigmoid Activation**: Logistic sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ with trainable biases across all non-input layers.
- **Online Backpropagation**: Pattern-by-pattern gradient descent with:
  - Configurable learning rate ($\eta$, default `0.1`).
  - Momentum coefficient ($\mu$, default `0.9`) to smooth weight updates and avoid local minima.
- **Early Stopping & Best Weights Tracking**: Automatically stops training if the training MSE fails to improve by more than $10^{-5}$ over 50 consecutive epochs. Restores the historical best-performing weight checkpoint upon termination.
- **Data Normalization (`-n`)**: Min-Max scaling feature that normalizes input attributes to $[-1, 1]$ and targets to $[0, 1]$.
- **Statistical Multi-Seed Evaluation**: Automatically evaluates performance over 5 distinct random seeds (`1, 2, 3, 4, 5`) to report mean and standard deviation of Mean Squared Error (MSE).
- **Weight Serialization**: Ability to save (`-w`) and load pre-trained network topologies and weight matrices.

---

## Project Architecture

```text
.
├── Makefile                     # Build automation script
├── la1.cpp                      # Main CLI driver and evaluation harness
├── mc/                          # Core neural network library
│   ├── MultilayerPerceptron.h   # MLP class definitions and layer structures
│   ├── MultilayerPerceptron.cpp # Feed-forward, backpropagation, and weight routines
│   ├── util.h                   # Dataset structures and mathematical helpers
│   └── util.cpp                 # Dataset parsing, normalizers, and stats
├── dat/                         # Benchmark and real-world datasets
│   ├── train_xor.dat / test_xor.dat
│   ├── train_sin.dat / test_sin.dat
│   ├── train_bikes.dat / test_bikes.dat
│   └── train_buoy_46025.dat / test_buoy_46025.dat
├── bin/                         # Compiled binaries directory (la1)
├── tests.py                     # Python runner for grid-search across architectures
├── grafica_convergencia.py      # Generates training/test loss convergence graphs
├── grafica_seno.py              # Visualizes sine wave regression fit
└── memoria/                     # Comprehensive LaTeX report and documentation
```

---

## Prerequisites & Build

### Requirements
- **C++ Compiler**: GCC (`g++`) or Clang supporting C++11 or higher.
- **Make**: GNU Make.
- **Python 3** *(Optional, for plots & test automation)*: `matplotlib`, `pandas`, `numpy`.

### Compilation

Compile the project using the provided `Makefile`:

```bash
# Build the executable (placed in bin/la1)
make

# Clean build artifacts
make clean

# Run default test configuration
make run
```

---

## Usage & Command-Line Options

The CLI program `./bin/la1` accepts the following arguments:

| Flag | Parameter | Description | Default |
| :--- | :--- | :--- | :--- |
| `-t` | `<path>` | **(Required)** Path to the training dataset file | — |
| `-T` | `<path>` | **(Required)** Path to the test dataset file | — |
| `-i` | `<int>` | Maximum number of training epochs / iterations | `1000` |
| `-l` | `<int>` | Number of hidden layers | `1` |
| `-h` | `<int>` | Number of neurons per hidden layer | `5` |
| `-e` | `<float>` | Learning rate ($\eta$) | `0.1` |
| `-m` | `<float>` | Momentum coefficient ($\mu$) | `0.9` |
| `-n` | — | Enable Min-Max normalization (inputs to $[-1, 1]$, targets to $[0, 1]$) | `false` |
| `-w` | `<path>` | Path to save/load network weights | Optional |
| `-p` | — | Prediction mode (outputs Kaggle-compatible CSV format) | `false` |
| `-s` / `--csv` | `<path>` | Save final summary metrics (Mean and SD) to a CSV file | Optional |

---

## Datasets

The repository includes four datasets in `dat/`:

1. **XOR (`train_xor.dat`, `test_xor.dat`)**:
   - Classical non-linearly separable classification problem.
   - Formatted with bipolar inputs $\{-1, 1\}$ and binary targets $\{0, 1\}$.
2. **Seno (`train_sin.dat`, `test_sin.dat`)**:
   - 1D continuous sinusoidal curve approximation $y = \sin(x)$.
3. **Bikes (`train_bikes.dat`, `test_bikes.dat`)**:
   - Real-world multivariate regression dataset predicting hourly bike-sharing demand from meteorological and temporal variables. Requires `-n`.
4. **Buoy 46025 / SWH (`train_buoy_46025.dat`, `test_buoy_46025.dat`)**:
   - Time series regression predicting Significant Wave Height (SWH) recorded by oceanic meteorological buoys. Requires `-n`.

---

## Execution Examples

### 1. XOR Problem
```bash
./bin/la1 -t dat/train_xor.dat -T dat/test_xor.dat -l 1 -h 8 -i 1000
```

### 2. Sine Wave Function Approximation
```bash
./bin/la1 -t dat/train_sin.dat -T dat/test_sin.dat -l 2 -h 8 -i 1000
```

### 3. Bike Sharing Regression (Normalized)
```bash
./bin/la1 -t dat/train_bikes.dat -T dat/test_bikes.dat -n -l 2 -h 32 -i 1000
```

### 4. Significant Wave Height (Buoy 46025, Normalized)
```bash
./bin/la1 -t dat/train_buoy_46025.dat -T dat/test_buoy_46025.dat -n -l 2 -h 64 -i 1000
```

---

## Automation & Plotting Scripts

Helper scripts are available in the root folder:

- **Run Architecture Benchmark**:
  ```bash
  python3 tests.py
  ```
  Executes grid searches across layer depths ($l \in \{1, 2\}$) and layer sizes ($h \in \{2, 4, 8, 16, 32, 64, 100\}$), producing summary CSV reports.

- **Generate Convergence Plots**:
  ```bash
  python3 grafica_convergencia.py
  ```
  Exports epoch-by-epoch loss convergence graphs (MSE vs. Iteration) to `graficos/`.

- **Generate Sine Function Comparison Plot**:
  ```bash
  python3 grafica_seno.py
  ```
  Plots ground truth targets against the network's predicted outputs.

---

## Report & Documentation

The directory [`memoria/`](file:///home/aramos03/Escritorio/MC/P01/memoria) includes the full technical report written in LaTeX, covering mathematical foundations, empirical ablation studies, convergence diagrams, and architectural analysis.

