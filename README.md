# Numerical Optimization for Large Scale Problems - Assignment

This repository contains the implementation of the final project for the **Numerical Optimization for Large Scale Problems** course (A.Y. 2025/2026) at Politecnico di Torino.

The project focuses on solving unconstrained large-scale optimization problems using second-order derivative-based methods, comparing their performance in terms of efficiency, accuracy, and robustness.

## 🚀 Project Overview

The assignment explores **Derivative-based Optimization** (Assignment 2.1) applied to high-dimensional test problems. The goal is to minimize objective functions where the dimension $n$ scales up to $10^5$.

### Implemented Methods
We have implemented the following optimization algorithms with **Back-tracking Line Search** using the Armijo condition:
1.  **Modified Newton Method:** Utilizes a Cholesky decomposition with diagonal correction ($\tau$-adjustment) to ensure a descent direction, exploiting the **banded structure** of the Hessian for computational efficiency.
2.  **Truncated Newton Method (Newton-CG):** An inexact Newton approach that solves the Newton system using the Conjugate Gradient method, suitable for very large-scale problems where explicit Hessian inversion is impractical.

### Test Problems
Following the assignment instructions, the solvers are tested on two problems:
* **Broyden Tridiagonal Function**
* **Banded Trigonometric Function**

Numerical derivatives via **Finite Differences** are also implemented to study the impact of gradient and Hessian approximations (with increments $h=10^{-k}$ and $h_i=10^{-k}|x_i|$) on convergence rates.

## 📂 Repository Structure

* `root/`
    * `methods.py`: Core implementation of Modified Newton and Truncated Newton algorithms.
    * `broyden.py`: Definition of the Broyden Tridiagonal problem including function, gradient, and sparse Hessian.
    * `banded_trig.py`: Definition of the Banded Trigonometric problem.
    * `utils.py`: Utility functions for backtracking line search, convergence analysis, and plotting.
    * `main.py`: Main execution script to run benchmarks across different dimensions ($n=2, 10^3, 10^4, 10^5$).
* `results/`: CSV files containing the numerical results of the tests.
* `latex_tables/`: Automatically generated LaTeX code for the report tables (Templates 1 and 2).
* `plots/`: Visualization of convergence rates and function top-views for $n=2$.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.10+
* NumPy
* SciPy
* Matplotlib

### Running the Optimization
To run the standard benchmark suite, execute:
```bash
python root/main.py
```
The script uses a random seed equal to the minimum student ID of the team members as required by the guidelines.

## 📊 Key Features
* **Sparsity Exploitation:** The Modified Newton method handles banded matrices to significantly reduce computational complexity.
* **Numerical Derivative Analysis:** Support for both constant increment $h$ and coordinate-dependent increment $h_i = 10^{-k}|\hat{x}_i|$ to evaluate approximation errors.
* **Comprehensive Logging:** Tracks iteration counts, objective function values, gradient norms, and execution time for performance comparison.

## 👥 Authors
* **Lucio Baiocchi** 
* **Leonardo Passafiume**

## 📜 License
This project is for educational purposes as part of the Numerical Optimization course requirements at Politecnico di Torino.
