#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for Q coefficients generation
"""
import numpy as np

from qmat.utils import checkOverriding, storeClass, importAll
from qmat.lagrange import LagrangeApproximation

class QGenerator(object):

    @classmethod
    def getInstance(cls):
        try:
            return cls()
        except TypeError:
            return cls(**cls.DEFAULT_PARAMS)

    @property
    def nodes(self):
        raise NotImplementedError("mouahahah")

    @property
    def Q(self):
        raise NotImplementedError("mouahahah")

    @property
    def weights(self):
        raise NotImplementedError("mouahahah")

    @property
    def weightsEmbedded(self):
        """
        These weights can be used to construct a secondary lower order method from the same stages.
        """
        raise NotImplementedError("no embedded weights implemented for {type(self).__name__}")

    @property
    def nNodes(self):
        return self.nodes.size

    @property
    def rightIsNode(self):
        return self.nodes[-1] == 1.

    @property
    def T(self):
        """Transfer matrix from zero-to-nodes to node-to-node"""
        M = self.Q.shape[0]
        T = np.eye(M)
        T[1:,:-1][np.diag_indices(M-1)] = -1
        return T

    @property
    def S(self):
        Q = np.asarray(self.Q)
        M = self.Q.shape[0]
        T = np.eye(M)
        T[1:,:-1][np.diag_indices(M-1)] = -1
        return T @ Q

    @property
    def Tinv(self):
        """Transfer matrix from node-to-node to zero-to-node"""
        M = self.Q.shape[0]
        return np.tri(M)

    @property
    def hCoeffs(self):
        approx = LagrangeApproximation(self.nodes)
        return approx.getInterpolationMatrix([1]).ravel()

    def genCoeffs(self, withS=False, hCoeffs=False, embedded=False):
        out = [self.nodes, self.weights, self.Q]

        if embedded:
            out[1] = np.vstack([out[1], self.weightsEmbedded])
        if withS:
            out.append(self.S)
        if hCoeffs:
            out.append(self.hCoeffs)
        return out

    @property
    def order(self):
        raise NotImplementedError("mouahahah")

    @property
    def orderEmbedded(self):
        return self.order - 1

    def solveDahlquist(self, lam, u0, T, nSteps, useEmbeddedWeights=False):
        nodes, weights, Q = self.nodes, self.weights, self.Q

        if useEmbeddedWeights:
            weights = self.weightsEmbedded

        uNum = np.zeros(nSteps+1, dtype=complex)
        uNum[0] = u0

        dt = T/nSteps
        A = np.eye(nodes.size) - lam*dt*Q
        for i in range(nSteps):
            b = np.ones(nodes.size)*uNum[i]
            uStages = np.linalg.solve(A, b)
            uNum[i+1] = uNum[i] + lam*dt*weights.dot(uStages)

        return uNum

    def errorDahlquist(self, lam, u0, T, nSteps, uNum=None, useEmbeddedWeights=False):
        if uNum is None:
            uNum = self.solveDahlquist(lam, u0, T, nSteps, useEmbeddedWeights=useEmbeddedWeights)
        times = np.linspace(0, T, nSteps+1)
        uExact = u0 * np.exp(lam*times)
        return np.linalg.norm(uNum-uExact, ord=np.inf)


Q_GENERATORS = {}

def register(cls:QGenerator)->QGenerator:
    # Check for correct overriding
    for name in ["nodes", "Q", "weights", "order"]:
        checkOverriding(cls, name)
    # Check that TEST_PARAMS are given and valid if no default constructor
    try:
        cls()
    except TypeError:
        try:
            params = cls.DEFAULT_PARAMS
        except AttributeError:
            raise AttributeError(
                f"{cls.__name__} requires DEFAULT_PARAMS attribute"
                " since it has no default constructor")
        try:
            cls(**params)
        except:
            raise TypeError(
                f"{cls.__name__} could not be instantiated with DEFAULT_PARAMS")
    # Store class (and aliases)
    storeClass(cls, Q_GENERATORS)
    return cls

def genQCoeffs(qType, withS=False, hCoeffs=False, embedded=False, **params):
    try:
        Generator = Q_GENERATORS[qType]
    except KeyError:
        raise ValueError(f"{qType=!r} is not available")
    gen = Generator(**params)
    return gen.genCoeffs(withS, hCoeffs, embedded)


# Import all local submodules
__all__ = importAll(locals(), __path__, __name__, __import__)
