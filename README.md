# Multilayer Perceptron (MLP) in C++

A modular, high-performance implementation of a **Multilayer Perceptron (MLP)** artificial neural network built from scratch in C++. The network supports both **Online** and **Offline (Batch)** Backpropagation with momentum, Xavier/Glorot weight initialization, early stopping, configurable output activations (Logistic Sigmoid and Softmax), and modular loss functions (MSE and Cross-Entropy) evaluated via function pointers.

Developed for the *Computational Models* (*Modelos Computacionales*) course at the **University of Córdoba (UCO)**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Prerequisites & Build](#prerequisites--build)
- [Usage & Command-Line Options](#usage--command-line-options)
- [Dataset Format & Benchmarks](#dataset-format--benchmarks)
- [Execution Examples](#execution-examples)

---

## Overview

This project implements a fully configurable feed-forward Multilayer Perceptron adapted for both classification tasks (e.g., non-linearly separable logic gates like XOR, binary classification such as ProPublica COMPAS, and multi-class benchmarks like noMNIST and categorized Bikes) and regression tasks.

The implementation includes:
- **Online and Offline (Batch) learning**: In offline mode, gradients are accumulated across the entire training set and averaged ($\frac{1}{N}$) before updating weights with momentum.
- **Probabilistic classification**: Softmax activation in the output layer paired with Categorical Cross-Entropy loss.
- **Analytic gradient optimizations**: Efficient $\mathcal{O}(n_H)$ error backpropagation across all combinations of activations (Sigmoid, Softmax) and loss functions (MSE, Cross-Entropy).
- **Classification Correctness Rate (CCR)**: Evaluation metric for classification tasks alongside standard continuous error metrics.

---

## Key Features

- **Flexible Architecture**:
  - $H \ge 2$ layers (1 input layer, $L \ge 1$ hidden layers of uniform width $h$, and 1 output layer).
  - Layer dimensions automatically match the input/output attributes of the loaded dataset.
- **Xavier / Glorot Weight Initialization**:
  - Synaptic weights uniformly sampled from $[-\frac{1}{\sqrt{n_{h-1}}}, \frac{1}{\sqrt{n_{h-1}}}]$, preventing vanishing/exploding gradients in wide networks.
- **Activation Functions**:
  - *Hidden layers*: Logistic Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$.
  - *Output layer*: Configurable between Logistic Sigmoid and numerically stable Softmax ($\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$).
- **Modular Loss Functions via Function Pointers**:
  - Mean Squared Error (MSE).
  - Cross-Entropy Loss with logarithmic lower clipping ($o_{po} \ge 10^{-10}$) to ensure numerical stability.
  - Dispatch table (`getErrorFunction(int code)`) selecting member function pointers without branch overhead.
- **Complete Gradient Formulations**:
  - Analytic simplifications for Softmax + Cross-Entropy ($\delta_j = out_j - d_j$) and Sigmoid + Cross-Entropy ($\delta_j = -d_j(1 - out_j)$).
  - Full Jacobian formulation for Softmax + MSE factored into linear time $\mathcal{O}(n_H)$.
- **Flexible Training Modes**:
  - *Online / Stochastic Mode* (`-o`): Weights updated pattern-by-pattern.
  - *Offline / Batch Mode* (default): Gradients accumulated over the entire epoch, divided by $N$, and updated once per epoch.
- **Convergence Control & Checkpointing**:
  - Momentum coefficient ($\mu$, default `1.0`) and learning rate ($\eta$, default `0.7`).
  - Early stopping: stops if training error does not decrease by at least $10^{-5}$ for 50 consecutive epochs.
  - Best-weights restoration: automatically restores the checkpoint with lowest training error before final test evaluation.
- **Data Normalization (`-n`)**:
  - Scales input features to $[-1, 1]$ while preserving target classes intact for classification.
  - Uses training minimum and maximum values for scaling test data to prevent information leakage.
- **Statistical Multi-Seed Evaluation**:
  - Evaluates over 5 seeds (`1, 2, 3, 4, 5`) reporting Mean $\pm$ Standard Deviation for training/test error and CCR.
- **Weight Serialization & Kaggle Prediction Mode**:
  - Saves best model architecture and weights (`-w`).
  - Prediction mode (`-p`) producing standard submission output format (`Id,Category`).

---

## Project Structure

```text
.
├── Makefile                     # Build automation script
├── la1.cpp                      # Main CLI driver, arguments parser, and test harness
├── mc/                          # Core neural network library
│   ├── MultilayerPerceptron.h   # MLP, Layer, and Neuron class definitions
│   ├── MultilayerPerceptron.cpp # Feed-forward, backpropagation, and training routines
│   ├── util.h                   # Dataset structures and mathematical utility headers
│   └── util.cpp                 # Dataset parsing, normalizers, and Softmax
└── README.md                    # Project documentation
```

---

## Prerequisites & Build

### Requirements
- **C++ Compiler**: GCC (`g++`) or Clang supporting C++11 or higher.
- **Build System**: GNU Make.

### Compilation

Compile the project using the provided `Makefile`:

```bash
# Build the executable (generates bin/la2 and symlink bin/la1)
make

# Clean build artifacts
make clean

# Run default execution configuration
make run
```

---

## Usage & Command-Line Options

The program accepts options in any order:

```bash
# Display help and usage instructions
./bin/la2 --help

# Run training and evaluation
./bin/la2 -t <train_file> [options]
```

| Flag | Parameter | Description | Default |
| :--- | :--- | :--- | :--- |
| `-t` | `<path>` | **(Required)** Path to the training dataset file | — |
| `-T` | `<path>` | Path to the test dataset file (defaults to training file if omitted) | Same as `-t` |
| `-i` | `<int>` | Maximum number of training epochs / iterations | `1000` |
| `-l` | `<int>` | Number of hidden layers | `1` |
| `-h` | `<int>` | Number of neurons per hidden layer | `4` |
| `-e` | `<float>` | Learning rate ($\eta$) | `0.7` |
| `-m` | `<float>` | Momentum coefficient ($\mu$) | `1.0` |
| `-o` | — | Enable Online training mode (default is Offline / Batch) | `false` |
| `-f` | `<0\|1>` | Error function: `0` for MSE, `1` for Cross-Entropy | `0` (MSE) |
| `-s` | — | Use Softmax activation on the output layer (default is Sigmoid) | `false` |
| `-n` | — | Normalize input features to $[-1, 1]$ | `false` |
| `-w` | `<path>` | Path to save the best model weights or load them in prediction mode | Optional |
| `-p` | — | Kaggle prediction mode (reads test file and prints `Id,Category`) | `false` |
| `-c` / `--csv` | `<path>` | Append final metrics (Mean $\pm$ SD) to a CSV file | Optional |
| `--help` | — | Display command-line usage information and exit | — |

---

## Dataset Format & Benchmarks

Datasets are formatted as plain text files with the following structure:
```text
<nOfInputs> <nOfOutputs> <nOfPatterns>
<x_11> <x_12> ... <x_1I> <d_11> <d_12> ... <d_1K>
<x_21> <x_22> ... <x_2I> <d_21> <d_22> ... <d_2K>
...
```

Typical classification benchmarks used with this project:
- **XOR**: Non-linearly separable logic function (2 inputs, 2 outputs with 1-of-2 encoding).
- **ProPublica COMPAS**: Fairness and recidivism risk assessment benchmark (binary classification).
- **Bikes**: Hourly demand level classification into 4 discretized traffic categories (multi-class).
- **noMNIST**: Character font image classification across 10 classes A–J (multi-class, 1-of-10 encoding).

---

## Execution Examples

### 1. XOR Classification (Default Settings: Offline, Sigmoid, MSE)
```bash
./bin/la2 -t dat/train_xor.dat -T dat/test_xor.dat -i 1000 -l 1 -h 4 -e 0.7 -m 1.0
```

### 2. ProPublica COMPAS with Softmax and Cross-Entropy
```bash
./bin/la2 -t dat/train_compas.dat -T dat/test_compas.dat -i 1000 -l 1 -h 8 -e 0.7 -m 1.0 -s -f 1 -n
```

### 3. noMNIST Multi-Class Classification
```bash
./bin/la2 -t dat/train_nomnist.dat -T dat/test_nomnist.dat -i 500 -l 1 -h 16 -e 0.5 -m 0.9 -s -f 1 -n
```

### 4. Online Training Mode with Weight Saving
```bash
./bin/la2 -t dat/train_xor.dat -T dat/test_xor.dat -o -i 500 -l 1 -h 4 -w weights.txt
```

### 5. Kaggle Prediction Mode
```bash
./bin/la2 -T dat/test_nomnist.dat -p -w weights.txt > submission.csv
```
The output format follows the standard Kaggle submission:
```csv
Id,Category
0,7
1,3
2,0
...
```

### 6. Logging Results to CSV
```bash
./bin/la2 -t dat/train_bikes.dat -T dat/test_bikes.dat -i 1000 -l 1 -h 16 -s -f 1 -n -c results.csv
```
Appends row with format:
`nCapasOcultas,nNeuronas,mediaErrorTrain,stdErrorTrain,mediaErrorTest,stdErrorTest,mediaCCRTrain,stdCCRTrain,mediaCCRTest,stdCCRTest`.
