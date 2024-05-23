#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule for QDelta coefficients based on algebraic approach
"""
import numpy as np
import scipy.linalg as spl

from qmat.qdelta import QDeltaGenerator, register

@register
class Exact(QDeltaGenerator):
    """Take Q (exact approximation)"""
    aliases = ["EXACT"]

    def getQDelta(self, k=None):
        np.copyto(self.QDelta, self.Q)
        return self.QDelta

@register
class PIC(QDeltaGenerator):
    """Picard approximation (zeros)"""
    aliases = ["Picard"]

    def getQDelta(self, k=None):
        return self.QDelta


@register
class LU(QDeltaGenerator):
    """LU approximation from Weiser"""

    def getQDelta(self, k=None):
        QT = self.Q.T
        _, _, U = spl.lu(QT)
        np.copyto(self.QDelta, U.T)
        return self.QDelta

@register
class LU2(QDeltaGenerator):
    """LU approximation from Weiser multiplied by 2"""

    def getQDelta(self, k=None):
        QT = self.Q.T
        _, _, U = spl.lu(QT)
        np.copyto(self.QDelta, U.T)
        self.QDelta *= 2
        return self.QDelta

@register
class QPar(QDeltaGenerator):
    """Approximation using diagonal of Q"""
    aliases = ["Qpar", "Qdiag"]

    def getQDelta(self, k=None):
        np.copyto(self.QDelta, np.diag(np.diag(self.Q)))
        return self.QDelta

@register
class QG(QDeltaGenerator):
    """Approximation using diagonal of Q"""
    aliases = ["GaussSeidel"]

    def getQDelta(self, k=None):
        np.copyto(self.QDelta, np.tril(self.Q))
        return self.QDelta
