#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for Q coefficients generation
"""
import numpy as np

from qmat.utils import checkOverriding, storeClass, importAll
from qmat.lagrange import LagrangeApproximation

class QGenerator(object):

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
    def S(self):
        Q = np.asarray(self.Q)
        M = self.Q.shape[0]
        T = np.eye(M)
        T[1:,:-1][np.diag_indices(M-1)] = -1
        return T @ Q

    @property
    def hCoeffs(self):
        approx = LagrangeApproximation(self.nodes)
        return approx.getInterpolationMatrix([1]).ravel()

    def genCoeffs(self, withS=False, hCoeffs=False):
        out = [self.nodes, self.weights, self.Q]
        if withS:
            out.append(self.S)
        if hCoeffs:
            out.append(self.hCoeffs)
        return out


    def solveDahlquist(self, lam, u0, T, nSteps):
        nodes, weights, Q = self.nodes, self.weights, self.Q

        uNum = np.zeros(nSteps+1, dtype=complex)
        uNum[0] = u0

        dt = T/nSteps
        A = np.eye(nodes.size) - lam*dt*Q
        for i in range(nSteps):
            b = np.ones(nodes.size)*uNum[i]
            uStages = np.linalg.solve(A, b)
            uNum[i+1] = uNum[i] + lam*dt*weights.dot(uStages)

        return uNum

    def errorDahlquist(self, lam, u0, T, nSteps, uNum=None):
        if uNum is None:
            uNum = self.solveDahlquist(lam, u0, T, nSteps)
        times = np.linspace(0, T, nSteps+1)
        uExact = u0 * np.exp(lam*times)
        return np.linalg.norm(uNum-uExact, ord=np.inf)


Q_GENERATORS:dict[str:QGenerator] = {}

def register(cls:QGenerator)->QGenerator:
    for name in ["nodes", "Q", "weights"]:
        checkOverriding(cls, name)
    storeClass(cls, Q_GENERATORS)
    return cls

def genQCoeffs(qType, withS=False, hCoeffs=False, **params):
    try:
        Generator = Q_GENERATORS[qType]
    except KeyError:
        raise ValueError(f"qType={qType} is not available")
    gen = Generator(**params)
    return gen.genCoeffs(withS, hCoeffs)


# Import all local submodules
__all__ = importAll(locals(), __path__, __name__, __import__)
