#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
**Main sub-packages** 📦

- :class:`qcoeff` : generates the :math:`Q`-coefficients (Butcher tables)
- :class:`qdelta` : generates :math:`Q_\Delta` approximations for :math:`Q` matrices

**Internal features** ⚙️

- :class:`lagrange` : barycentric polynomial approximations (integral, interpolation, derivative)
- :class:`nodes` : generates multiple types of quadrature nodes
- :class:`utils` : sub-package for utilities (functions & classes)

Examples
--------
>>> from qmat import genQCoeffs, genQDeltaCoeffs
>>>
>>> # Coefficients of a specific collocation method
>>> nodes, weights, Q = genQCoeffs(
>>>     "Collocation", nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
>>>
>>> # QDelta matrix from Implicit-Euler based SDC
>>> QDelta = genQDeltaCoeffs("IE", nodes=nodes)
>>>
>>> # Butcher table of the classical explicit RK4 method
>>> c, b, A = genQCoeffs("ERK4")

>>> from qmat import Q_GENERATORS, QDELTA_GENERATORS
>>> print(Q_GENERATORS)         # list all available generator classes for Q coefficients
>>> print(QDELTA_GENERATORS)    # list all available generator classes for QDelta approximations
"""
from qmat.qcoeff import genQCoeffs, Q_GENERATORS
from qmat.qdelta import genQDeltaCoeffs, QDELTA_GENERATORS

__all__ = [
    "genQCoeffs", "genQDeltaCoeffs",
    "Q_GENERATORS", "QDELTA_GENERATORS"]
