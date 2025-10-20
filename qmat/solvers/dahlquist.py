#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule containing various solvers for the Dahlquist equation that can be used with `qmat`-generated coefficients.
"""
import numpy as np


class Dahlquist():

    def __init__(self, lam, u0=1, T=1, nSteps=1):
        self.u0 = u0
        self.T = T
        self.nSteps = nSteps
        self.dt = T/nSteps

        self.lam = np.asarray(lam)
        try:
            lamU = self.lam*u0
        except:
            raise ValueError("error when computing lam*u0")
        self.uShape = tuple(lamU.shape)
        self.uDtype = lamU.dtype

    @staticmethod
    def checkCoeff(Q, weights):
        Q = np.asarray(Q)
        nNodes = Q.shape[0]
        assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

        if weights is not None:
            weights = np.asarray(weights)
            assert weights.ndim == 1, "weights must be a 1D vector"
            assert weights.size == nNodes, "weights size is not the same as the node size"
        else:
            assert np.allclose(Q.sum(axis=1)[-1], 1), "last node must be 1 if weights are not given"

        return nNodes, Q, weights


    def solve(self, Q, weights):
        nNodes, Q, weights = self.checkCoeff(Q, weights)

        # Collocation problem matrix
        A = np.eye(nNodes) - self.lam[..., None, None]*self.dt*Q

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.uDtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):
            b = np.ones(nNodes)*uNum[i][..., None]
            uNodes = np.linalg.solve(A, b[..., None])[..., 0]
            if weights is not None:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lamI[..., None]*uNodes, weights)
            else:
                uNum[i+1] = uNodes[..., -1]

        return uNum

    @staticmethod
    def checkCoeffSDC(Q, weights, QDelta, nSweeps):
        Q = np.asarray(Q)
        nodes = Q.sum(axis=1)
        nNodes = nodes.size
        assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

        if weights is not None:
            weights = np.asarray(weights)
            assert weights.ndim == 1, "weights must be a 1D vector"
            assert weights.size == nNodes, "weights size is not the same as the node size"
        else:
            assert np.allclose(nodes[-1], 1), "last node must be 1 if weights are not given"

        QDelta = np.asarray(QDelta)
        if QDelta.ndim == 3:
            assert QDelta.shape == (nSweeps, nNodes, nNodes), "inconsistent shape for QDelta"
        else:
            assert QDelta.shape == (nNodes, nNodes), "inconsistent shape for QDelta"
            QDelta = np.repeat(QDelta[None, ...], nSweeps, axis=0)

        return nNodes, Q, weights, QDelta, nSweeps

    def solveSDC(self, Q, weights, QDelta, nSweeps):
        nNodes, Q, weights, QDelta, nSweeps = self.checkCoeffSDC(Q, weights, QDelta, nSweeps)

        # Preconditioner for each sweeps
        P = np.eye(nNodes)[None, ...] \
            - self.lam[..., None, None, None]*self.dt*QDelta

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.uDtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):

            uNodes = np.ones(nNodes)*uNum[i][..., None]
            uNodes = uNodes[..., :, None]   # shape [..., nNodes, 1]

            for k in range(nSweeps):

                b = uNum[i][..., None, None] \
                    + self.lam[..., None, None]*self.dt*(Q - QDelta[k]) @ uNodes

                # b has shape [..., nNodes, 1]
                # P[k] has shape [..., nNodes, nNodes]
                # output has shape [..., nNodes, 1]
                uNodes = np.linalg.solve(P[..., k, :, :], b)

            uNodes = uNodes[..., :, 0]  # back to shape [..., nNodes]

            if weights is None:
                uNum[i+1] = uNodes[..., -1]
            else:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lam[..., None]*uNodes, weights)

        return uNum


class DahlquistIMEX():

    def __init__(self, lamI, lamE, u0=1, T=1, nSteps=1):
        self.u0 = u0
        self.T = T
        self.nSteps = nSteps
        self.dt = T/nSteps

        self.lamI = np.asarray(lamI)
        self.lamE = np.asarray(lamE)
        try:
            lamU = (self.lamI + self.lamE)*u0
        except:
            raise ValueError("error when computing (lamI + lamE)*u0")
        self.uShape = tuple(lamU.shape)
        self.uDtype = lamU.dtype


    @staticmethod
    def checkCoeff(QI, wI, QE, wE):
        QI, QE = np.asarray(QI), np.asarray(QE)
        nodes = QI.sum(axis=1)
        assert np.allclose(nodes, QE.sum(axis=1)), "QI and QE do not correspond to the same nodes"

        nNodes = QI.shape[0]
        assert QI.shape == (nNodes, nNodes), "QI is not a square matrix"
        assert QI.shape == QE.shape, "QI and QE do not have the same shape"

        useWeights = True
        if wI is None or wE is None:
            assert wE is None and wI is None, "it's either weights for everyone or no weight"
            useWeights = False

        return nNodes, QI, wI, QE, wE, useWeights


    def solve(self, QI, wI, QE, wE):
        nNodes, QI, wI, QE, wE, useWeights = self.checkCoeff(QI, wI, QE, wE)

        # Collocation problem matrix
        A = np.eye(nNodes) \
            - self.lamI[..., None, None]*self.dt*QI \
            - self.lamE[..., None, None]*self.dt*QE

        # Solution vector for each time-step
        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.uDtype)
        uNum[0] = self.u0

        # Time-stepping loop
        for i in range(self.nSteps):

            b = np.ones(nNodes)*uNum[i][..., None]
            uNodes = np.linalg.solve(A, b[..., None])[..., 0]

            if useWeights:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lamI[..., None]*uNodes, wI)
                uNum[i+1] += self.dt*np.dot(self.lamE[..., None]*uNodes, wE)
            else:
                uNum[i+1] = uNodes[..., -1]

        return uNum


    @staticmethod
    def checkCoeffSDC(Q, weights, QDeltaI, QDeltaE, nSweeps):
        Q = np.asarray(Q)
        nodes = Q.sum(axis=1)
        nNodes = nodes.size
        assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

        if weights is not None:
            weights = np.asarray(weights)
            assert weights.ndim == 1, "weights must be a 1D vector"
            assert weights.size == nNodes, "weights size is not the same as the node size"

        QDeltaI = np.asarray(QDeltaI)
        QDeltaE = np.asarray(QDeltaE)
        if QDeltaI.ndim == 3:
            assert QDeltaI.shape == (nSweeps, nNodes, nNodes), "inconsistent shape for QDeltaI"
        else:
            assert QDeltaI.shape == (nNodes, nNodes), "inconsistent shape for QDeltaE"
            QDeltaI = np.repeat(QDeltaI[None, ...], nSweeps, axis=0)
        if QDeltaE.ndim == 3:
            assert QDeltaE.shape == (nSweeps, nNodes, nNodes), "inconsistent shape for QDeltaE"
        else:
            assert QDeltaE.shape == (nNodes, nNodes), "inconsistent shape for QDeltaE"
            QDeltaE = np.repeat(QDeltaE[None, ...], nSweeps, axis=0)

        return nNodes, Q, weights, QDeltaI, QDeltaE, nSweeps


    def solveSDC(self, Q, weights, QDeltaI, QDeltaE, nSweeps):
        nNodes, Q, weights, QDeltaI, QDeltaE, nSweeps = self.checkCoeffSDC(Q, weights, QDeltaI, QDeltaE, nSweeps)

        # Preconditioner for each sweeps
        P = np.eye(nNodes)[None, ...] \
            - self.lamI[..., None, None, None]*self.dt*QDeltaI \
            - self.lamE[..., None, None, None]*self.dt*QDeltaE

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.uDtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):

            uNodes = np.ones(nNodes)*uNum[i][..., None]
            uNodes = uNodes[..., :, None]   # shape [..., nNodes, 1]

            for k in range(nSweeps):

                b = uNum[i][..., None, None] \
                    + self.lamI[..., None, None]*self.dt*(Q - QDeltaI[k]) @ uNodes \
                    + self.lamE[..., None, None]*self.dt*(Q - QDeltaE[k]) @ uNodes

                # b has shape [..., nNodes, 1]
                # P[k] has shape [..., nNodes, nNodes]
                # output has shape [..., nNodes, 1]
                uNodes = np.linalg.solve(P[..., k, :, :], b)

            uNodes = uNodes[..., :, 0]  # back to shape [..., nNodes]

            if weights is None:
                uNum[i+1] = uNodes[..., -1]
            else:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(
                    (self.lamI[..., None] + self.lamE[..., None])*uNodes, weights)

        return uNum
