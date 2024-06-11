QMat Package
************

`qmat` is a python package to generate matrix coefficients related to Collocation methods, Spectral Deferred Corrections (SDC), 
and more general multi-stages time-integration methods (like Runge-Kutta, etc ...).

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

and many different **lower-triangular** approximation of the :math:`Q` matrix, named :math:`Q_\Delta`,
which are key elements for Spectral Deferred Correction (SDC), or more general Iterated Runge-Kutta Methods.


    📜 *If you are already familiar with those concepts, you can use this package like this :*


.. code-block:: python

    from qmat import genQCoeffs, genQDeltaCoeffs

    # Coefficients or specific collocation method
    nodes, weights, Q = genQCoeffs("Collocation", nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")

    # QDelta matrix from Implicit-Euler based SDC
    QDelta = genQDeltaCoeffs("IE", nodes=nodes)

    # Butcher table of the classical explicit RK4 method
    c, b, A = genQCoeffs("ERK4")


*But if you are new to this, then welcome ! ... and please have a look at the notebook tutorials* 😉

Doc Contents
============

.. toctree::
    :maxdepth: 1

    installation
    notebooks
    contributing
    API reference <autoapi/qmat/index>

Links
=====

* Code repository: https://github.com/Parallel-in-Time/qmat
* Documentation: http://qmat.readthedocs.org

Developer
=========

* `Thibaut Lunet <https://github.com/tlunet/>`_