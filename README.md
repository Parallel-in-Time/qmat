# QMat project

[![CI pipeline for qmat](https://github.com/Parallel-in-Time/qmat/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/Parallel-in-Time/qmat/actions/workflows/ci_pipeline.yml)
[![codecov](https://codecov.io/gh/Parallel-in-Time/qmat/graph/badge.svg?token=MO0LDVH5NN)](https://codecov.io/gh/Parallel-in-Time/qmat)

`qmat` is a python package to generate matrix coefficients related to Collocation methods, Spectral Deferred Corrections (SDC), 
and more general multi-stages time-integration methods (like Runge-Kutta, etc ...).

It allows to generate $Q$-coefficients for multi-stages methods(equivalent to Butcher tables) :

$$
Q\text{-coefficients : }
\begin{array}
    {c|c}
    \tau & Q \\
    \hline\\[-1em]
    & w^\top
\end{array}
\quad \Leftrightarrow \quad
\begin{array}
    {c|c}
    c & A \\
    \hline\\[-1em]
    & b^\top
\end{array}
\quad\text{(Butcher table)}
$$
and many different **lower-triangular** approximation of the $Q$ matrix, named $Q_\Delta$.
Those $Q_\Delta$ matrices are key elements for SDC, or more general Iterated Runge-Kutta Methods.

## Installation

:hammer_and_wrench: In construction ...

## Basic usage

:scroll: _If you are already familiar with those concepts, you can use this package like this :_

```python
from qmat import genQCoeffs, genQDeltaCoeffs

# Coefficients or specific collocation method
nodes, weights, Q = genQCoeffs("Collocation", nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")

# QDelta matrix from Implicit-Euler based SDC
QDelta = genQDeltaCoeffs("IE", nodes=nodes)

# Butcher table of the classical explicit RK4 method
c, b, A = genQCoeffs("ERK4")
```

> :bell: _If you are not familiar with SDC or related methods, and want to learn more about it, checkout the [**extended documentation and tutorials ...**](./tutorials/)_ (:hammer_and_wrench: in construction)

