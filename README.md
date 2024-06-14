# QMat Package

[![CI pipeline for qmat](https://github.com/Parallel-in-Time/qmat/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/Parallel-in-Time/qmat/actions/workflows/ci_pipeline.yml)
[![codecov](https://codecov.io/gh/Parallel-in-Time/qmat/graph/badge.svg?token=MO0LDVH5NN)](https://codecov.io/gh/Parallel-in-Time/qmat)

`qmat` is a python package to generate matrix coefficients related to Collocation methods, Spectral Deferred Corrections (SDC), 
and more general multi-stages time-integration methods (like Runge-Kutta, etc ...).

It allows to generate $Q$-coefficients for multi-stages methods (equivalent to Butcher tables) :

$$
Q\text{-coefficients : }
\begin{array}{c|c}
\tau & Q \\
\hline
\phantom{\tau} & w^\top
\end{array}
\quad \Leftrightarrow \quad
\begin{array}{c|c}
c & A \\
\hline
\phantom{\tau} & b^\top
\end{array}
\quad\text{(Butcher table)}
$$

and many different **lower-triangular** approximation of the $Q$ matrix, named $Q_\Delta$,
which are key elements for Spectral Deferred Correction (SDC), or more general Iterated Runge-Kutta Methods.



## Installation

🛠️ Still in construction, only installation from source is enable yet, see [current instructions ...](./docs/installation.md)

## Basic usage

📜 _If you are already familiar with those concepts, you can use this package like this :_

```python
from qmat import genQCoeffs, genQDeltaCoeffs

# Coefficients or specific collocation method
nodes, weights, Q = genQCoeffs("Collocation", nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")

# QDelta matrix from Implicit-Euler based SDC
QDelta = genQDeltaCoeffs("IE", nodes=nodes)

# Butcher table of the classical explicit RK4 method
c, b, A = genQCoeffs("ERK4")
```

> 🔔 _If you are not familiar with SDC or related methods, and want to learn more about it, checkout the 
> [latest documentation build](https://qmat.readthedocs.io/en/latest/) and 
in particular the [**step by step tutorials**](https://qmat.readthedocs.io/en/latest/notebooks.html)_


For any contribution, please checkout out (very cool) [Contribution Guidelines](./docs/contributing.md)

