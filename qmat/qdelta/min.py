#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule for QDelta coefficients based on minimization approaches
"""
import warnings
import numpy as np
import scipy.optimize as spo

from qmat.qdelta import QDeltaGenerator, register
import qmat.qdelta.mincoeffs as tables
from qmat.qcoeff.collocation import Collocation


@register
class MIN(QDeltaGenerator):
    aliases = ["MIN-Speck"]

    def rho(self, x):
        M = self.size
        return max(abs(
            np.linalg.eigvals(np.eye(M) - np.diag(x).dot(self.Q))))

    def computeQDelta(self, k=None):
        x0 = 10 * np.ones(self.size)
        d = spo.minimize(self.rho, x0, method='Nelder-Mead')
        return np.linalg.inv(np.diag(d.x))


class FromTable(QDeltaGenerator):

    def __init__(self, nNodes, nodeType, quadType, **kwargs):
        self.nNodes = nNodes
        self.nodeType = nodeType
        self.quadType = quadType
    
    @property
    def size(self):
        return self.nNodes

    def computeQDelta(self, k=None):
        name = self.__class__.__name__
        try:
            table = getattr(tables, name)
            coeffs = table[self.nodeType][self.quadType][self.size]
            coeffs = np.asarray(coeffs, dtype=float)
            assert coeffs.ndim == 1
            assert coeffs.size == self.size
        except KeyError:
            raise NotImplementedError(
                f"no {name} MIN coefficients for"
                f"{self.nNodes} {self.nodeType} {self.quadType} nodes")
        except AssertionError:
            raise ValueError(
                f"MIN coefficients stored for {name} are inconsistent : {coeffs}")
        except:
            raise ValueError(
                f"could not convert {name} MIN coefficients to numpy array")
        return np.diag(coeffs)

    def check(cls):
        try:
            getattr(tables, cls.__name__)
        except AttributeError:
            raise AttributeError(
                f"no MIN coefficients table found for {cls.__name__}"
                " in qmat.qdelta.mincoeffs")
        return cls

def registerTable(cls:FromTable)->FromTable:
    return register(FromTable.check(cls))


@registerTable
class MIN3(FromTable):
    """
    These values have been obtained using Indie Solver, a commercial solver for
    black-box optimization which aggregates several state-of-the-art optimization
    methods (free academic subscription plan).
    Objective function :
        sum over 17^2 values of lamdt, real and imaginary
    (WORKS SURPRISINGLY WELL!)
    """
    aliases = ["Magic_Numbers"]


@registerTable
class MIN_VDHS(FromTable):
    aliases = ["VDHS"]


@register
class MIN_SR_NS(QDeltaGenerator):
    aliases = ["MIN-SR-NS", "MIN_GT"]

    def __init__(self, nodes, **kwargs):
        self.nodes = np.asarray(nodes)

    @property
    def size(self):
        return self.nodes.size

    def computeQDelta(self, k=None):
        return np.diag(self.nodes)/self.size


@register
class MIN_SR_S(QDeltaGenerator):
    aliases = ["MIN-SR-S"]

    def __init__(self, nNodes, nodeType, quadType, **kwargs):
        self.nodeType = nodeType
        self.quadType = quadType
        self.coll = Collocation(nNodes, nodeType, quadType)

    @property
    def size(self):
        return self.coll.nodes.size

    def computeCoeffs(self, M, a=None, b=None):
        """
        Compute diagonal coefficients for a given number of nodes M.
        If `a` and `b` are given, then it uses as initial guess:

        >>> a * nodes**b / M

        If `a` is not given, then do not care about `b` and uses as initial guess:

        >>> nodes / M

        Parameters
        ----------
        M : int
            Number of collocation nodes.
        a : float, optional
            `a` coefficient for the initial guess.
        b : float, optional
            `b` coefficient for the initial guess.

        Returns
        -------
        coeffs : array
            The diagonal coefficients.
        nodes : array
            The nodes associated to the current coefficients.
        """
        nodeType, quadType = self.nodeType, self.quadType
        if M == self.size:
            collM = self.coll
        else:
            collM = Collocation(nNodes=M, nodeType=nodeType, quadType=quadType)
        QM, nodesM = collM.Q, collM.nodes

        if quadType in ['LOBATTO', 'RADAU-LEFT']:
            QM = QM[1:, 1:]
            nodesM = nodesM[1:]
        nCoeffs = len(nodesM)

        if nCoeffs == 1:
            coeffs = np.diag(QM)

        else:

            def nilpotency(coeffs):
                """Function verifying the nilpotency from given coefficients"""
                coeffs = np.asarray(coeffs)
                kMats = [(1 - z) * np.eye(nCoeffs) + z * np.diag(1 / coeffs) @ QM for z in nodesM]
                vals = [np.linalg.det(K) - 1 for K in kMats]
                return np.array(vals)

            if a is None:
                coeffs0 = nodesM / M
            else:
                coeffs0 = a * nodesM**b / M

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = spo.fsolve(nilpotency, coeffs0, xtol=1e-15)

        # Handle first node equal to zero
        if quadType in ['LOBATTO', 'RADAU-LEFT']:
            coeffs = np.asarray([0.0] + list(coeffs))
            nodesM = np.asarray([0.0] + list(nodesM))

        return coeffs, nodesM

    @staticmethod
    def fit(coeffs, nodes):
        """Function fitting given coefficients to a power law"""

        def lawDiff(ab):
            a, b = ab
            return np.linalg.norm(a * nodes**b - coeffs)

        sol = spo.minimize(lawDiff, [1.0, 1.0], method="nelder-mead")
        return sol.x

    def computeQDelta(self, k=None):
        # Compute coefficients incrementally
        a, b = None, None
        m0 = 2 if self.quadType in ['LOBATTO', 'RADAU-LEFT'] else 1
        for m in range(m0, self.size + 1):
            coeffs, nodes = self.computeCoeffs(m, a, b)
            if m > 1:
                a, b = self.fit(coeffs * m, nodes)
        return np.diag(coeffs)


@register
class MIN_SR_FLEX(MIN_SR_S):
    _K_DEP = True
    aliases = ["MIN-SR-FLEX"]

    def computeQDelta(self, k=None):
        if k is None:
            k = 1
        if k < 1:
            raise ValueError(f"k must be greater than 0 ({k})")
        if k <= self.size:
            return np.diag(self.coll.nodes/k)
        else:
            try:
                self._QDelta_MIN_SR_S
            except AttributeError:
                self._QDelta_MIN_SR_S = super().computeQDelta()
            return self._QDelta_MIN_SR_S
