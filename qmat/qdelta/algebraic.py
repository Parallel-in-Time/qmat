#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule for QDelta coefficients based on algebraic approach
"""
import numpy as np
import scipy.linalg as spl

from qmat.qdelta import QDeltaGenerator, register


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
    """LU approximation from Weiser"""

    def computeQDelta(self, k=None):
        _, _, U = spl.lu(self.Q.T)
        return U.T


@register
class LU2(QDeltaGenerator):
    """LU approximation from Weiser multiplied by 2"""

    def computeQDelta(self, k=None):
        _, _, U = spl.lu(self.Q.T)
        return 2*U.T


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
