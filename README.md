# Numerical Optimization for Large Scale Problems

> **Academic Report:** Assignment 2.1 - Derivative-based Optimization

This repository contains the source code, numerical results, and analysis for the implementation of second-order unconstrained optimization methods. The project focuses on assessing the efficiency, scalability, and robustness of these algorithms on high-dimensional problems (up to $N = 100,000$).

## 🎯 Objective
The primary goal is to implement and compare two advanced second-order optimization algorithms:
1. **Modified Newton Method**: Utilizes a Modified Cholesky Factorization to ensure descent directions in non-convex regions.
2. **Truncated Newton Method (Newton-CG)**: Approximates the Newton step using the Conjugate Gradient method, designed to handle large-scale problems without exact Hessian factorization.

Both methods are globalized using a **Backtracking Line Search** strategy.

## 📊 Test Problems
The solvers were benchmarked on two classic large-scale optimization problems:
* **Broyden Tridiagonal Function**: Features a symmetric and pentadiagonal Hessian (bandwidth $b=2$).
* **Banded Trigonometric Function**: Features a diagonal Hessian after algebraic decoupling.

## 🛠️ Implementation & Optimizations
To achieve optimal performance and linear time complexity $O(N)$, several problem-specific and structural optimizations were implemented:

* **Sparse Matrix Operations**: Heavy reliance on banded storage formats (`scipy.linalg.cholesky_banded`) and sparse diagonal matrices (`scipy.sparse.diags`) to reduce memory footprint and computational cost from $O(N^3)$ to $O(N)$.
* **Dynamic Hessian Perturbation ($\tau$)**: For the Modified Newton method, an iterative $\tau$-adjustment strategy ensures the minimal necessary perturbation to make the Hessian positive definite, preserving original curvature information.
* **Adaptive Forcing Sequence ($\eta_k$)**: For the Truncated Newton method, $\eta_k = \min(0.5, \sqrt{||\nabla F_k||})$ is used to loosely solve the system when far from the optimum and recover quadratic convergence near the solution.
* **Finite Differences & Graph Coloring**:
  * Implemented highly vectorized Finite Difference (FD) schemes for gradient and Hessian approximations.
  * Exploited **Graph Coloring (Stride Strategy)** to compute the exact Hessian via gradient differences in $O(1)$ evaluations (e.g., exactly 5 evaluations for the pentadiagonal problem).

## 📈 Key Findings
The experimental analysis up to dimension $N = 100,000$ revealed:
* **Modified Newton Superiority**: For highly structured sparsity (tridiagonal/diagonal), exploiting exact matrix factorizations yields significant performance gains. It completed the Broyden problem at $N=100,000$ in $\approx 0.45$ seconds.
* **Truncated Newton Limitations**: While generally preferred for its "matrix-free" low memory footprint, TN encountered numerical instability and early CG truncation on the Trigonometric problem due to negative curvature regions.
* **Finite Difference Efficiency**: Vectorized FD approximations introduced negligible computational overhead, offering results nearly indistinguishable from exact analytical derivatives when tuned correctly ($h=10^{-8}$).

## 👥 Authors
* **Leonardo Passafiume** (s358616) - Politecnico di Torino
* **Lucio Baiocchi** (s360244) - Politecnico di Torino

**Date:** January 12, 2026