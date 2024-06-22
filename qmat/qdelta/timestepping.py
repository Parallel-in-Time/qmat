#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule for QDelta coefficients based on time-stepping scheme
"""
import numpy as np

from qmat.qdelta import QDeltaGenerator, register


class TimeStepping(QDeltaGenerator):

    def __init__(self, nodes, tLeft=0, **kwargs):
        nodes = np.asarray(nodes)

        deltas = nodes.copy()
        deltas[0] = nodes[0] - tLeft
        deltas[1:] = np.ediff1d(nodes)
        
        self.deltas = deltas
        self.nodes = nodes
        self.tLeft = tLeft

    @property
    def size(self):
        return self.nodes.size


@register
class BE(TimeStepping):
    """Approximation based on Backward Euler steps between the nodes"""
    aliases = ["IE"]

    def computeQDelta(self, k=None):
        QDelta, M, deltas = self.zeros, self.nodes.size, self.deltas
        for i in range(M):
            QDelta[i:, :M-i] += np.diag(deltas[:M-i])
        return QDelta


@register
class FE(TimeStepping):
    """Approximation based on Forward Euler steps between the nodes"""
    aliases = ["EE"]

    def computeQDelta(self, k=None):
        QDelta, M, deltas = self.zeros, self.nodes.size, self.deltas
        for i in range(1, M):
            QDelta[i:, :M-i] += np.diag(deltas[1:M-i+1])
        return QDelta

    @property
    def dTau(self):
        return self.nodes*0 + self.deltas[0]


@register
class TRAP(TimeStepping):
    """Approximation based on Trapezoidal Rule between the nodes"""
    aliases = ["CN"]

    def computeQDelta(self, k=None):
        QDelta, M, deltas = self.zeros, self.nodes.size, self.deltas
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

    def computeQDelta(self, k=None):
        return np.diag(self.nodes) - self.tLeft


@register
class TRAPAR(TimeStepping):
    """Approximation based on parallel Trapezoidal Rule from zero to nodes"""

    def computeQDelta(self, k=None):
        return np.diag(self.nodes/2) - self.tLeft

    @property
    def dTau(self):
        return self.nodes/2.0 - self.tLeft
