# Tutorials

📜 *Extensive user guide, mostly based on step-by-step tutorials ...*

All tutorials are written in jupyter notebooks, that can be :

- read using the [online documentation](https://qmat.readthedocs.io/en/latest/notebooks.html)
- downloaded from the [notebook folder](https://github.com/Parallel-in-Time/qmat/tree/main/docs/notebooks) and played with 

> 🛠️ Basic usage tutorials are finalized and polished, the rest is still in construction ... 

Notebooks are categorized into three main sections :

1. **Basic usage** : how to generate and use basic $Q$-coefficients and $Q_\Delta$ approximations, through a step-by-step tutorial going from generic Runge-Kutta methods to SDC for simple problems. 
2. **Extended usage** : description of the additional features or `qmat`, like the $S$-matrix, 
the `hCoeffs` and `dTau` coefficients, ... going deeper into SDC 


```{eval-rst}
Base usage
==========

📜 *From Butcher Tables to Spectral Deferred Corrections*

.. toctree::
    :maxdepth: 1
    :glob:

    notebooks/0*

Extended usage
==============

📜 *Going deeper into SDC's understanding*

.. toctree::
    :maxdepth: 1
    :glob:

    notebooks/1*
