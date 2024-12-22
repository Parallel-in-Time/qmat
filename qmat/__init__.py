#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
**Main sub-packages** 📦

- :class:`qcoeff` : to generate the :math:`Q`-coefficients (Butcher tables)
- :class:`qdelta` : to generate :math:`Q_\Delta` approximations for :math:`Q` matrices

**Utility modules** ⚙️

- :class:`lagrange` : Barycentric polynomial approximations (integral, interpolation, derivation)
- :class:`nodes` : generation of multiple types of quadrature nodes
- :class:`sdc` : utility function to run SDC on simple problems
- :class:`mathutils` : utility functions for math operations
- :class:`utils` : utility functions for the whole package

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
