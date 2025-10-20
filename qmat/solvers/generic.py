#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule containing various generic solvers that can be used with `qmat`-generated coefficients.
"""
import numpy as np
import scipy.optimize as sco
from scipy.linalg import blas

from qmat.solvers.dahlquist import Dahlquist


class LinearMultiNode():

    def __init__(self, u0, evalF, fSolve=None, tEnd=1, nSteps=1, t0=0):
        u0 = np.asarray(u0)
        if u0.size < 1e3:
            self.innerSolver = sco.fsolve
        else:
            self.innerSolver = sco.newton_krylov
        self.u0 = u0
        self.t0 = t0
        self.tEnd = tEnd
        self.nSteps = nSteps
        self.dt = (tEnd-t0)/nSteps

        try:
            uEval = np.zeros_like(u0)
            evalF(u=u0, t=t0, out=uEval)
        except:
            raise ValueError("evalF cannot be properly evaluated into an array like u0")
        self.evalF = evalF

        if fSolve is not None:
            self.fSolve = fSolve
        try:
            dt = 1e-1
            uEval *= -dt
            uEval += u0
            uSolve = np.copy(u0)
            uSolve += 1e-3*np.linalg.norm(uSolve, np.inf)
            self.fSolve(a=dt, rhs=uEval, t=t0, out=uSolve)
        except:
            raise ValueError("fSolve cannot be properly evaluated into an array like u0")
        np.testing.assert_allclose(
            uSolve, u0, err_msg="fSolve does not satisfy the fixed-point problem with u0",
            atol=1e-15)

        self.axpy = blas.get_blas_funcs('axpy', dtype=self.dtype)

    @property
    def uShape(self):
        return self.u0.shape

    @property
    def dtype(self):
        return self.u0.dtype

    def evalF(self, u:np.ndarray, t:float, out:np.ndarray):
        raise NotImplementedError("evalF must be provided")


    def fSolve(self, a:float, rhs:np.ndarray, t:float, out:np.ndarray):
        """
        Solve u - a*f(u, t) = rhs using out as initial guess and storing the final solution into it
        """

        def func(u:np.ndarray):
            """compute res = u - a*f(u,t) - rhs"""
            u = u.reshape(self.uShape)
            res = np.empty_like(u)
            self.evalF(u, t, out=res)
            res *= -a
            res += u
            res -= rhs
            return res.ravel()

        sol = self.innerSolver(func, out.ravel()).reshape(self.uShape)
        np.copyto(out, sol)


    @staticmethod
    def lowerTri(Q:np.ndarray):
        return np.allclose(np.triu(Q, k=1), np.zeros(Q.shape))


    def solve(self, Q, weights, uNum=None):
        nNodes, Q, weights = Dahlquist.checkCoeff(Q, weights)

        assert self.lowerTri(Q), "lower triangular matrix Q expected for non-linear solver"
        Q, weights = self.dt*Q, self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        fEvals = np.zeros((nNodes, *self.uShape), dtype=self.dtype)

        times = np.linspace(self.t0, self.tEnd, self.nSteps+1)
        tau = Q.sum(axis=1)

        # time-stepping loop
        for i in range(self.nSteps):
            uNode = uNum[i+1]
            np.copyto(uNode, uNum[i])

            # loop on nodes (stages)
            for m in range(nNodes):
                tNode = times[i]+tau[m]

                # build RHS
                np.copyto(rhs, uNum[i])
                for j in range(m):
                    self.axpy(a=Q[m, j], x=fEvals[j], y=rhs)

                # solve node (if non-zero diagonal coefficient)
                if Q[m, m] != 0:
                    self.fSolve(a=Q[m, m], rhs=rhs, t=tNode, out=uNode)
                else:
                    np.copyto(uNode, rhs)

                # evalF on current store stage
                self.evalF(u=uNode, t=tNode, out=fEvals[m])

            # step update (if not, uNum[i+1] is already the last stage)
            if weights is not None:
                uNum[i+1] = uNum[i]
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fEvals[m], y=uNum[i+1])

        return uNum


    def solveSDC(self, Q, weights, QDelta, nSweeps, uNum=None):
        nNodes, Q, weights, QDelta, nSweeps = Dahlquist.checkCoeffSDC(Q, weights, QDelta, nSweeps)

        for qDelta in QDelta:
            assert self.lowerTri(qDelta), "lower triangular matrices QDelta expected for non-linear SDC solver"
        Q, QDelta = self.dt*Q, self.dt*QDelta
        if weights is not None:
            weights = self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        fEvals = [np.zeros((nNodes, *self.uShape), dtype=self.dtype)
                  for _ in range(2)]

        times = np.linspace(self.t0, self.tEnd, self.nSteps+1)
        tau = Q.sum(axis=1)

        # time-stepping loop
        for i in range(self.nSteps):

            # copy initialization
            self.evalF(u=uNum[i], t=times[i], out=fEvals[0][0])
            np.copyto(fEvals[0][1:], fEvals[0][0])

            uNode = uNum[i+1]

            # loop on sweeps (iterations)
            for k in range(nSweeps):
                np.copyto(uNode, uNum[i])

                fK0 = fEvals[0]
                fK1 = fEvals[1]
                qDelta = QDelta[k]

                # loop on nodes (stages)
                for m in range(nNodes):
                    tNode = times[i] + tau[m]

                    # initialize RHS
                    np.copyto(rhs, uNum[i])

                    # add quadrature terms
                    for j in range(nNodes):
                        self.axpy(a=Q[m, j], x=fK0[j], y=rhs)

                    # add correction terms (from previous nodes)
                    for j in range(m):
                        self.axpy(a= qDelta[m, j], x=fK1[j], y=rhs)
                        self.axpy(a=-qDelta[m, j], x=fK0[j], y=rhs)

                    # diagonal term (current node)
                    if qDelta[m, m] != 0:
                        self.axpy(a=-qDelta[m, m], x=fK0[m], y=rhs)
                        self.fSolve(a=qDelta[m, m], rhs=rhs, t=tNode, out=uNode)
                    else:
                        np.copyto(uNode, rhs)

                    # evalF on current node
                    self.evalF(u=uNode, t=tNode, out=fK1[m])

                # invert fK0 and fK1 for the next sweep
                fEvals[0], fEvals[1] = fEvals[1], fEvals[0]

            # step update (if not, uNum[i+1] is already the last stage)
            if weights is not None:
                uNum[i+1] = uNum[i]
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fK1[m], y=uNum[i+1])

        return uNum


class GenericMultiNode(LinearMultiNode):

    def __init__(self, u0, evalF, nodes, fSolve=None, tEnd=1, nSteps=1, t0=0):
        super().__init__(u0, evalF, fSolve, tEnd, nSteps, t0)
        self.nodes = np.asarray(nodes, dtype=float)

    @property
    def nNodes(self):
        return self.nodes.size
    nStages = nNodes


    def evalPsi(self, uVals, fEvals, out, t0=0):
        raise NotImplementedError(
            "specialized Integrator must implement its evalPsi method")

    def nodeSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
        """solve u-psi(u, u0, fEvals) = rhs"""

        def func(u:np.ndarray):
            u = u.reshape(self.uShape)
            res = np.empty_like(u)
            self.evalPsi([*uPrev, u], fEvals, out=res, t0=t0)
            res *= -1
            res += u
            res -= rhs
            return res.ravel()

        sol = self.innerSolver(func, out.ravel()).reshape(self.uShape)
        np.copyto(out, sol)


    def stepUpdate(self, u0, uNodes, fEvals, out):
        np.copyto(out, uNodes[-1])
        fEvals[0], fEvals[-1] = fEvals[-1], fEvals[0]


    def solve(self, uNum=None):
        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        uNodes = np.zeros((self.nNodes, *self.uShape), dtype=self.dtype)
        fEvals = [np.zeros(self.uShape, dtype=self.dtype)
                  for _ in range(self.nNodes+1)]
        self.evalF(uNum[0], self.t0, out=fEvals[0])

        times = np.linspace(self.t0, self.tEnd, self.nSteps+1)
        tau = self.dt*self.nodes

        # time-stepping loop
        for i in range(self.nSteps):

            # initialize first node with starting value for step
            np.copyto(uNodes[0], uNum[i])

            # loop on nodes
            for m in range(self.nNodes):
                self.nodeSolve(
                    [uNum[i], *uNodes[:m]], fEvals[:m+1], out=uNodes[m], t0=times[i])
                self.evalF(u=uNodes[m], t=times[i]+tau[m], out=fEvals[m+1])

            # step update
            self.stepUpdate(uNum[i], uNodes, fEvals, out=uNum[i+1])


        return uNum


    def solveSDC(self, Q, weights, nSweeps, uNum=None):

        Q = self.dt*Q

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        uNodes = [np.zeros((self.nNodes, *self.uShape), dtype=self.dtype)
                  for _ in range(2)]
        fEvals = [[np.zeros(self.uShape, dtype=self.dtype)
                   for _ in range(self.nNodes+1)]
                  for _ in range(2)]

        times = np.linspace(self.t0, self.tEnd, self.nSteps+1)
        tau = self.dt*self.nodes

        # time-stepping loop
        for i in range(self.nSteps):

            # copy initialization
            np.copyto(uNodes[0], uNum[i])
            self.evalF(uNum[i], t=times[i], out=fEvals[0])
            np.copyto(fEvals[1:], fEvals[0])

            uTmp = uNum[i+1]

            # loop on sweeps (iterations)
            for k in range(nSweeps):

                uK0 = uNodes[0]
                uK1 = uNodes[1]

                # loop on nodes (stages)
                for m in range(self.nNodes):

                    # initialize RHS
                    np.copyto(rhs, uNum[i])

                    # add quadrature terms
                    for j in range(self.nNodes):
                        self.axpy(a=Q[m, j], x=fEvals[j], y=rhs)

                    # substract k correction term
                    if k == 0:
                        self.axpy(a=-tau[m], x=fEvals[0], y=rhs)
                        rhs -= uNum[i]
                    else:
                        self.evalPsi(uNum[i], *uK0[:m+1], out=uTmp, t0=times[i])
                        rhs -= uTmp

                    # solve with k+1 correction
                    self.nodeSolve(
                        uNum[i], *uK1[:m], out=uK1[m], rhs=rhs, t0=times[i])

                # compute f evals
                for m in range(self.nNodes):
                    self.evalF(uK1[m], t=times[i]+tau[m], out=fEvals[m])

                # invert uK0 and uK1 for next sweep
                uNodes.rotate()

            # step update (copy of last node solution per default)
            self.stepUpdate(*uK1, out=uNum[i+1], t0=times[i])

        return uNum



class ForwardEuler(GenericMultiNode):

    def evalPsi(self, uVals, fEvals, out, t0=0):
        m = len(uVals) - 1
        assert m > 0
        assert len(fEvals) == m

        tau = [t0] + (t0 + self.dt*self.nodes).tolist()

        # u0 + dt1 f0 + dt2 f1 + ... + dtm f{m-1}
        np.copyto(out, uVals[0])
        for i in range(m):
            self.axpy(a=tau[i+1]-tau[i], x=fEvals[i], y=out)


    def nodeSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
        self.evalPsi([*uPrev, out], fEvals, out, t0=t0)
        out += rhs


class BackwardEuler(GenericMultiNode):

    def evalPsi(self, uVals, fEvals, out, t0=0):
        m = len(uVals) - 1
        assert m > 0
        assert len(fEvals) == m

        tau = [t0] + (t0 + self.dt*self.nodes).tolist()

        # dtm f{m} + ... + dt2 f2 + dt1 f1 + u0
        self.evalF(uVals[-1], tau[m+1], out=out)
        for i in range(m-1):
            self.axpy(a=tau[i+1]-tau[i], x=fEvals[i+1], y=out)
        out += uVals[0]
