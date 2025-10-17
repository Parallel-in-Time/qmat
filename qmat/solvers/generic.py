#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule containing various generic solvers that can be used with `qmat`-generated coefficients.
"""
import numpy as np
import scipy.optimize as sco
from scipy.linalg import blas

from collections import deque

from qmat.solvers.dahlquist import Dahlquist


class NonLinear():

    DEFAULT_FSOLVE = sco.fsolve

    def __init__(self, u0, evalF, fSolve=None, T=1, nSteps=1):
        self.u0 = np.asarray(u0)
        if self.u0.size > 1e3:
            self.DEFAULT_FSOLVE = sco.newton_krylov
        self.axpy = blas.get_blas_funcs('axpy', dtype=self.uDtype)
        
        self.T = T
        self.nSteps = nSteps
        self.dt = T/nSteps
        
        try:
            uOut = np.zeros_like(u0)
            uEval = evalF(u=u0, t=0, out=uOut)
        except:
            raise ValueError("evalF cannot be properly evaluated into an array like u0")
        assert uOut is uEval, "evalF output is not its out argument"
        self.evalF = evalF
        
        if fSolve is not None:
            self.fSolve = fSolve
        try:
            uEval *= -1
            uEval += u0
            uOut = np.zeros_like(u0)
            uSolve = fSolve(a=1, b=uEval, uInit=u0, t=0, out=uOut)
        except:
            raise ValueError("fSolve cannot be properly evaluated into an array like u0")
        assert uOut is uSolve, "fSolve output is not its out argument"
        np.testing.assert_allclose(uSolve, u0, err_msg="fSolve does not satisfy the fixed-point problem with u0")
        

    @property
    def uShape(self):
        return self.u0.shape
    
    @property
    def uDtype(self):
        return self.u0.dtype

    def evalF(self, u, t, out):
        raise NotImplementedError("very weird error ...")


    def fSolve(self, a, b, uInit, t, out):
        """
        Solve u - a*evalF(u, t) = b using uInit as initial guess and storing u into out
        """
        np.copyto(out, self.DEFAULT_FSOLVE(lambda u: u - a*self.evalF(u, t) - b, uInit))


    @staticmethod
    def lowerTri(Q:np.ndarray):
        return np.allclose(np.triu(Q, k=1), np.zeros(Q.shape))

    
    def solve(self, Q, weights, uNum=None):
        nNodes, Q, weights = Dahlquist.checkCoeff(Q, weights)
        
        assert self.lowerTri(Q), "lower triangular matrix Q expected"
        Q, weights = self.dt*Q, self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.uDtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.uDtype)
        fEvals = np.zeros((nNodes, *self.uShape), dtype=self.uDtype)

        times = np.linspace(0, self.T, self.nSteps+1)
        tau = Q.sum(axis=1)

        # time-stepping loop
        for i in range(self.nSteps):
            np.copyto(uNum[i+1], uNum[i])
            uStage = uNum[i+1]

            # stages loop
            for m in range(nNodes):
                tStage = times[i]+tau[m]
            
                # build RHS
                np.copyto(rhs, uNum[i])
                for j in range(m):
                    self.axpy(a=Q[m, j], x=fEvals[j], y=rhs)

                # solve stage (if non-zero diagonal coefficient)
                if Q[m, m] != 0:
                    self.fSolve(a=Q[m, m], b=rhs, uInit=uStage, t=tStage, out=uStage)
                else:
                    np.copyto(uStage, rhs)

                # eval and store stage
                self.evalF(u=uStage, t=tStage, out=fEvals[m])

            # step update (if not, uNum[i+1] is already the last stage)
            if weights is not None:
                uNum[i+1] = uNum[i]
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fEvals[m], y=uNum[i+1])

        return uNum
    

    def solveSDC(self, Q, weights, QDelta, nSweeps, uNum=None):
        nNodes, Q, weights, QDelta, nSweeps = Dahlquist.checkCoeffSDC(Q, weights, QDelta, nSweeps)

        for qDelta in QDelta:
            assert self.lowerTri(qDelta), "lower triangular matrices QDelta expected"
        Q, QDelta, weights = self.dt*Q, self.dt*QDelta, self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.uDtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.uDtype)
        fEvals = deque([np.zeros_like(rhs) for _ in range(2)])

        times = np.linspace(0, self.T, self.nSteps+1)
        tau = Q.sum(axis=1)

        # time-stepping loop
        for i in range(self.nSteps):
            np.copyto(uNum[i+1], uNum[i])
            uNode = uNum[i+1]
            
            # copy initialization
            self.evalF(u=uNum[i], t=times[i], out=fK0[0])
            for m in range(1, nNodes):
                np.copyto(fK0[m], fK0[0])  

            # loop on sweeps
            for k in range(nSweeps):

                fK0 = fEvals[0]
                fK1 = fEvals[1]
                qDelta = QDelta[k]

                # loop on nodes
                for m in range(nNodes):
                    tNode = times[i]+tau[m]

                    # initialize RHS
                    np.copyto(rhs, uNum[i])

                    # add quadrature terms
                    for j in range(m):
                        self.axpy(a=Q[m, j], x=fK0[j], y=rhs)

                    # add correction terms (from previous nodes)
                    for j in range(m):
                        self.axpy(a= qDelta[m, j], x=fK1[j], y=rhs)
                        self.axpy(a=-qDelta[m, j], x=fK0[j], y=rhs)

                    # diagonal term (current node)
                    if qDelta[m, m] != 0:
                        self.axpy(a=-qDelta[m, m], x=fK0[m], y=rhs)
                        self.fSolve(a=qDelta[m, m], b=rhs, uInit=uNode, t=tNode, out=uNode)
                    else:
                        np.copyto(uNode, rhs)

                    # evalF on node
                    self.evalF(u=uNode, t=tNode, out=fK1[m])

                # invert fK0 and fK1 for the next sweep
                fEvals.rotate()

            # step update (if not, uNum[i+1] is already the last stage)
            if weights is not None:
                uNum[i+1] = uNum[i]
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fK1[m], y=uNum[i+1])

                
