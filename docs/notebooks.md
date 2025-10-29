# Notebook Tutorials

📜 *Extensive user guide, mostly based on step-by-step tutorials ...*

All tutorials are written in jupyter notebooks, that can be :

- read using the [online documentation](https://qmat.readthedocs.io/en/latest/notebooks.html)
- downloaded from the [notebook folder](https://github.com/Parallel-in-Time/qmat/tree/main/docs/notebooks) and played with

> 🛠️ Basic usage tutorials are finalized and polished, the rest is still in construction ...

Notebooks are categorized into those main sections :

1. **Basic usage** : how to generate and use basic $Q$-coefficients and $Q_\Delta$ approximations, through a step-by-step tutorial going from generic Runge-Kutta methods to SDC for simple problems.
2. **Extended usage** : additional features or `qmat` ($S$-matrix, `hCoeffs`, `dTau` coefficients, ...) to go deeper into SDC
3. **Components usage** : how to use the main utility modules, like `qmat.lagrange`, etc ...


```{eval-rst}
Base usage tutorial
===================

📜 *From Butcher Tables to Spectral Deferred Corrections ...*

.. toctree::
    :maxdepth: 1
    :glob:

    notebooks/0*

Advanced tutorials
==================

📜 *Going deeper into advanced time-integration topics ...*

.. toctree::
    :maxdepth: 1
    :glob:

    notebooks/1*

Components usage
================

📜 *How to use the utility modules ...*

.. toctree::
    :maxdepth: 1
    :glob:

    notebooks/2*
```
