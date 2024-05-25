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
        M = self.QDelta.shape[0]
        return max(abs(
            np.linalg.eigvals(np.eye(M) - np.diag(x).dot(self.Q))))

    def getQDelta(self, k=None):
        x0 = 10 * np.ones(self.Q.shape[0])
        d = spo.minimize(self.rho, x0, method='Nelder-Mead')
        np.copyto(self.QDelta, np.linalg.inv(np.diag(d.x)))
        return self.QDelta


class FromTable(QDeltaGenerator):

    def __init__(self, nNodes, nodeType, quadType, **kwargs):
        self.nNodes = nNodes
        self.nodeType = nodeType
        self.quadType = quadType
        self.QDelta = np.zeros((nNodes, nNodes), dtype=float)

    def getQDelta(self, k=None):
        name = self.__class__.__name__
        try:
            table = getattr(tables, name)
            coeffs = table[self.nodeType][self.quadType][self.nNodes]
        except KeyError:
            raise NotImplementedError(
                "no {name} MIN coefficients for"
                f"{self.nNodes} {self.nodeType} {self.quadType} nodes")
        np.copyto(self.QDelta, np.diag(coeffs))
        return self.QDelta

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

    def __init__(self, nodes):
        self.nodes = np.asarray(nodes)
        M = self.nodes.size
        self.QDelta = np.zeros((M, M), dtype=float)

    def getQDelta(self, k=None):
        np.copyto(self.QDelta, np.diag(self.nodes)/self.nodes.size)
        return self.QDelta


@register
class MIN_SR_S(QDeltaGenerator):
    aliases = ["MIN-SR-S"]

    def __init__(self, nNodes, nodeType, quadType, **kwargs):
        self.nNodes = nNodes
        self.nodeType = nodeType
        self.quadType = quadType
        self.QDelta = np.zeros((nNodes, nNodes), dtype=float)
        self.coll = Collocation(nNodes, nodeType, quadType)

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
        if M == self.nNodes:
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

    def getQDelta(self, k=None):
        # Compute coefficients incrementally
        a, b = None, None
        m0 = 2 if self.quadType in ['LOBATTO', 'RADAU-LEFT'] else 1
        for m in range(m0, self.nNodes + 1):
            coeffs, nodes = self.computeCoeffs(m, a, b)
            if m > 1:
                a, b = self.fit(coeffs * m, nodes)

        np.copyto(self.QDelta, np.diag(coeffs))
        return self.QDelta


@register
class MIN_SR_FLEX(MIN_SR_S):
    aliases = ["MIN-SR-FLEX"]

    def getQDelta(self, k=None):
        if k is None:
            k = 1
        if k < 1:
            raise ValueError(f"k must be greater than 0 ({k})")
        nodes = self.coll.nodes
        if k <= nodes.size:
            np.copyto(self.QDelta, np.diag(nodes)/k)
        else:
            try:
                self.QDelta_MIN_SR_S
            except AttributeError:
                self.QDelta_MIN_SR_S = super().getQDelta()
            np.copyto(self.QDelta, self.QDelta_MIN_SR_S)
        return self.QDelta
