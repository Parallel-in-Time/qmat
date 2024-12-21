QMat Package
************

.. raw:: html

    <a href="https://pypi.org/project/qmat/">
        <img alt="PyPI - Package" src="https://img.shields.io/pypi/v/qmat?logo=python">
    </a>
    <a href="https://pypistats.org/packages/qmat">
        <img alt="PyPI - Download" src="https://img.shields.io/pypi/dm/qmat?logo=pypi">
    </a>
    <a href="https://github.com/Parallel-in-Time/qmat">
        <img alt="Last Commit" src="https://img.shields.io/github/last-commit/parallel-in-time/qmat/main?logo=github" />
    </a>
    <a href="https://github.com/Parallel-in-Time/qmat/actions/workflows/ci_pipeline.yml">
        <img alt="CI pipeline" src="https://github.com/Parallel-in-Time/qmat/actions/workflows/ci_pipeline.yml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/Parallel-in-Time/qmat">
        <img alt="Codecov" src="https://codecov.io/gh/Parallel-in-Time/qmat/graph/badge.svg?token=MO0LDVH5NN">
    </a>

`qmat` is a python package to generate matrix coefficients related to Collocation methods, 
Spectral Deferred Corrections (SDC), and more generally for Runge-Kutta methods.

It allows to generate :math:`Q`-coefficients for multi-stages methods (equivalent to Butcher tables) :

.. math::

    Q\text{-coefficients : }
    \begin{array}
        {c|c}
        \tau & Q \\
        \hline
        & w^\top
    \end{array}
    \quad \Leftrightarrow \quad
    \begin{array}
        {c|c}
        c & A \\
        \hline
        & b^\top
    \end{array}
    \quad\text{(Butcher table)}

and many different **lower-triangular** approximations of the :math:`Q` matrix, named :math:`Q_\Delta`,
which are key elements for Spectral Deferred Correction (SDC), or more general Iterated Runge-Kutta Methods.

.. raw:: html

    <a href="https://zenodo.org/doi/10.5281/zenodo.11956478">
        <img alt="DOI" src="https://zenodo.org/badge/804826743.svg">
    </a>


This package can be installed using `pip` :

.. code-block:: bash

    pip install qmat

... but you can also use `conda` or installation from sources, see the :doc:`Installation Instructions 💾<installation>`


    📜 *If you are already familiar with those concepts, you can use this package like this :*

.. code-block:: python

    from qmat import genQCoeffs, genQDeltaCoeffs

    # Coefficients or specific collocation method
    nodes, weights, Q = genQCoeffs("Collocation", nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")

    # QDelta matrix from Implicit-Euler based SDC
    QDelta = genQDeltaCoeffs("IE", nodes=nodes)

    # Butcher table of the classical explicit RK4 method
    c, b, A = genQCoeffs("ERK4")


*But if you are new to this, then welcome ! ... and please have a look at the* **step by step tutorials** *below* 😉

    For any contribution, please checkout out (very cool) :doc:`Contribution Guidelines 🔑<contributing>`
    and the current :doc:`Development Roadmap 🎯<devdoc/roadmap>`


Projects relying on `qmat`
==========================

* `pySDC <https://github.com/Parallel-in-Time/pySDC>`_ : *Python implementation of the spectral deferred correction (SDC) approach and its flavors, esp. the multilevel extension MLSDC and PFASST.*


Doc Contents
============

.. toctree::
    :maxdepth: 1

    installation
    notebooks
    contributing
    API reference <api/qmat/index>

Links
=====

* Code repository: https://github.com/Parallel-in-Time/qmat
* Issues Tracker : https://github.com/Parallel-in-Time/qmat/issues
* Q&A : https://github.com/Parallel-in-Time/qmat/discussions/categories/q-a
* Project Proposals : https://github.com/Parallel-in-Time/qmat/discussions/categories/project-proposals

Developer
=========

* `Thibaut Lunet <https://github.com/tlunet/>`_