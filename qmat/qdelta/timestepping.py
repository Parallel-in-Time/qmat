#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule for QDelta coefficients based on time-stepping scheme
"""
import numpy as np

from qmat.qdelta import QDeltaGenerator, register

class TimeStepping(QDeltaGenerator):

    def __init__(self, nodes, **kwargs):
        self.nodes = np.asarray(nodes)
        deltas = nodes.copy()
        deltas[1:] = np.ediff1d(nodes)
        self.deltas = deltas
        M = self.nodes.size
        self.QDelta = np.zeros((M, M), dtype=float)


@register
class BE(TimeStepping):
    """Approximation based on Backward Euler steps between the nodes"""
    aliases = ["IE"]

    def getQDelta(self, k=None):
        QDelta, M, deltas = self.QDelta, self.nodes.size, self.deltas
        for i in range(M):
            QDelta[i:, :M-i] += np.diag(deltas[:M-i])
        return self.storeAndReturn(QDelta)


@register
class FE(TimeStepping):
    """Approximation based on Forward Euler steps between the nodes"""
    aliases = ["EE"]

    def getQDelta(self, k=None):
        QDelta, M, deltas = self.QDelta, self.nodes.size, self.deltas
        for i in range(1, M):
            QDelta[i:, :M-i] += np.diag(deltas[1:M-i+1])
        return self.storeAndReturn(QDelta)

    @property
    def dTau(self):
        return self.nodes*0 + self.deltas[0]

@register
class TRAP(TimeStepping):
    """Approximation based on Trapezoidal Rule between the nodes"""
    aliases = ["CN"]

    def getQDelta(self, k=None):
        QDelta, M, deltas = self.QDelta, self.nodes.size, self.deltas
        for i in range(0, M):
            QDelta[i:, :M-i] += np.diag(deltas[:M-i])
        for i in range(1, M):
            QDelta[i:, :M-i] += np.diag(deltas[1:M-i+1])
        QDelta /= 2.0
        return QDelta

    @property
    def dTau(self):
        return self.nodes*0 + self.deltas[0]/2.0


@register
class BEPAR(TimeStepping):
    """Approximation based on parallel Backward Euler steps from zero to nodes"""
    aliases = ["IEpar"]

    def getQDelta(self, k=None):
        self.QDelta[:] = np.diag(self.nodes)
        return self.QDelta


@register
class TRAPAR(TimeStepping):
    """Approximation based on parallel Trapezoidal Rule from zero to nodes"""

    def getQDelta(self, k=None):
        self.QDelta[:] = np.diag(self.nodes/2)
        return self.QDelta

    @property
    def dTau(self):
        return self.nodes/2.0
