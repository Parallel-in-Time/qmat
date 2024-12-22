#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule for QDelta coefficients based on algebraic approaches,
in particular the `LU trick` from `[Weiser, 2014]`_.

Examples
--------
>>> from qmat.qcoeff.collocation import Collocation
>>> coll = Collocation(nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
>>>
>>> from qmat import genQDeltaCoeffs
>>> QDelta = genQDeltaCoeffs("LU", Q=coll.Q)
>>>
>>> from qmat.qdelta.algebraic import LDU
>>> gen = LDU(Q=coll.Q)
>>> QDelta = gen.getQDelta()
"""
import numpy as np
import scipy.linalg as spl

from qmat.qdelta import QDeltaGenerator, register
from qmat.mathutils import lduFactorization


@register
class PIC(QDeltaGenerator):
    """Picard approximation (zeros)"""
    aliases = ["Picard"]

    def computeQDelta(self, k=None):
        return self.zeros


@register
class Exact(QDeltaGenerator):
    """Takes Q (exact approximation)"""
    aliases = ["EXACT"]

    def computeQDelta(self, k=None):
        return self.Q.copy()  # TODO: do we really want a copy here ... ?


@register
class LU(QDeltaGenerator):
    """LU approximation from `[Weiser, 2014] <https://link.springer.com/article/10.1007/s10543-014-0540-y>`_"""

    def computeQDelta(self, k=None):
        _, _, U = spl.lu(self.Q.T)
        return U.T


@register
class LU2(LU):
    """LU approximation from `[Weiser, 2014]`_ multiplied by 2"""

    def computeQDelta(self, k=None):
        return super().computeQDelta()*2


@register
class QPar(QDeltaGenerator):
    """Approximation using diagonal of Q"""
    aliases = ["Qpar", "Qdiag"]

    def computeQDelta(self, k=None):
        return np.diag(np.diag(self.Q))


@register
class GS(QDeltaGenerator):
    """Approximation using lower triangular part of Q"""
    aliases = ["GaussSeidel"]

    def computeQDelta(self, k=None):
        return np.tril(self.Q)


@register
class LDU(QDeltaGenerator):
    """Diagonal approximation using LDU factorization"""

    def computeQDelta(self, k=None):
        return lduFactorization(self.Q)[1]
