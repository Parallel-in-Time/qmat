#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module with simple usage SDC type time-stepping solvers
"""
import numpy as np


def solveDahlquistSDC(lam, u0, T, nSteps, nSweeps, Q, QDelta,
                      weights=None):
    uNum = np.zeros(nSteps+1, dtype=complex)
    uNum[0] = u0

    M = Q.shape[0]

    dt = T/nSteps
    P = np.eye(M) - lam*dt*QDelta
    for i in range(nSteps):

        uNodes = np.ones(M)*uNum[i]

        for k in range(nSweeps):
            b = uNum[i] + lam*dt*(Q-QDelta) @ uNodes
            uNodes = np.linalg.solve(P, b)

        if weights is not None:
            uNum[i+1] = uNum[i] + lam*dt*weights.dot(uNodes)
        else:
            uNum[i+1] = uNodes[-1]

    return uNum


def errorDahlquistSDC(lam, u0, T, nSteps, nSweeps, Q, QDelta,
                      weights=None, uNum=None):
    if uNum is None:
        uNum = solveDahlquistSDC(
            lam, u0, T, nSteps, nSweeps, Q, QDelta,
            weights=weights)

    times = np.linspace(0, T, nSteps+1)
    uExact = u0 * np.exp(lam*times)
    return np.linalg.norm(uNum-uExact, ord=np.inf)
