#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule containing various generic solvers that can be used with `qmat`-generated coefficients.
"""
import numpy as np
import scipy.optimize as sco
from scipy.linalg import blas

from qmat.solvers.dahlquist import Dahlquist
from qmat.lagrange import LagrangeApproximation


class Problem():

    def __init__(self, u0):
        u0 = np.asarray(u0)
        if u0.size < 1e3:
            self.innerSolver = sco.fsolve
        else:
            self.innerSolver = sco.newton_krylov

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

    def test(self, t0=0, dt=1e-1, eps=1e-3):
        u0 = self.u0

        try:
            uEval = np.zeros_like(u0)
            self.evalF(u=u0, t=t0, out=uEval)
        except:
            raise ValueError("evalF cannot be properly evaluated into an array like u0")

        try:
            dt = dt
            uEval *= -dt
            uEval += u0
            uSolve = np.copy(u0)
            uSolve += eps*np.linalg.norm(uSolve, np.inf)
            self.fSolve(a=dt, rhs=uEval, t=t0, out=uSolve)
        except:
            raise ValueError("fSolve cannot be properly evaluated into an array like u0")
        np.testing.assert_allclose(
            uSolve, u0, err_msg="fSolve does not satisfy the fixed-point problem with u0",
            atol=1e-15)


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
                np.copyto(uNum[i+1], uNum[i])
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fEvals[m], y=uNum[i+1])

        return uNum


    def solveSDC(self, nSweeps, Q, weights, QDelta, uNum=None):
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
                np.copyto(uNum[i+1], uNum[i])
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


    def evalPhi(self, uVals, fEvals, out, t0=0):
        raise NotImplementedError(
            "specialized Integrator must implement its evalPsi method")

    def nodeSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
        """solve u-psi(u, u0, fEvals) = rhs"""

        def func(u:np.ndarray):
            u = u.reshape(self.uShape)
            res = np.empty_like(u)
            self.evalPhi([*uPrev, u], fEvals, out=res, t0=t0)
            res *= -1
            res += u
            res -= rhs
            return res.ravel()

        sol = self.innerSolver(func, out.ravel()).reshape(self.uShape)
        np.copyto(out, sol)


    def stepUpdate(self, u0, uNodes, fEvals, out):
        """Update end-step solution and ensure that fEvals[0] contains its evaluation"""
        assert self.nodes[-1] == 1
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
                    [uNum[i], *uNodes[:m]], fEvals[:m+1], rhs=uNum[i], out=uNodes[m], t0=times[i])
                self.evalF(u=uNodes[m], t=times[i]+tau[m], out=fEvals[m+1])

            # step update
            self.stepUpdate(uNum[i], uNodes, fEvals, out=uNum[i+1])

        return uNum


    def solveSDC(self, nSweeps, Q=None, weights=None, uNum=None):

        if Q is None:
            approx = LagrangeApproximation(self.nodes)
            Q = approx.getIntegrationMatrix([(0, tau) for tau in self.nodes])
            if weights is True:
                weights = approx.getIntegrationMatrix([(0, 1)]).ravel()
            else:
                weights = None
        else:
            nNodes, Q, weights = Dahlquist.checkCoeff(Q, weights)

            assert nNodes == self.nNodes, "solver and Q do not have the same number of nodes"
            assert np.allclose(Q.sum(axis=1), self.nodes), "solver and Q do not have the same nodes"

        Q = self.dt*Q
        if weights is not None:
            weights = self.dt*weights

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
            self.evalF(uNum[i], self.t0, out=fEvals[0][0])
            np.copyto(fEvals[1][0], fEvals[0][0])   # u_0^{1} = u_0^{0}
            for m in range(self.nNodes):
                np.copyto(fEvals[0][m+1], fEvals[0][0])  # u_m^{k} = u_0^{0}

            uTmp = uNum[i+1]    # use next step as buffer for k correction term

            # loop on sweeps (iterations)
            for _ in range(nSweeps):

                uK0, uK1 = uNodes
                fK0, fK1 = fEvals

                # loop on nodes (stages)
                for m in range(self.nNodes):

                    # initialize RHS
                    np.copyto(rhs, uNum[i])

                    # add quadrature terms
                    fK = fK0[1:]  # note : ignore f(u0) term in fK0
                    for j in range(self.nNodes):
                        self.axpy(a=Q[m, j], x=fK[j], y=rhs)

                    # substract k correction term
                    self.evalPhi(
                        [uNum[i], *uK0[:m+1]], fK0[:m+2], out=uTmp, t0=times[i])
                    rhs -= uTmp

                    # solve with k+1 correction
                    self.nodeSolve(
                        [uNum[i], *uK1[:m]], fK1[:m+1], out=uK1[m], rhs=rhs, t0=times[i])

                    # evalF on k+1 node solution
                    self.evalF(uK1[m], t=times[i]+tau[m], out=fK1[m+1])

                # invert uK0/fK0 and uK1/fK1 for next sweep
                fEvals[0], fEvals[1] = fEvals[1], fEvals[0]
                uNodes[0], uNodes[1] = uNodes[1], uNodes[0]

            # step update
            if weights is not None:
                np.copyto(uNum[i+1], uNum[i])
                fK = fK1[1:]  # note : ignore f(u0) term in fK0
                for m in range(self.nNodes):
                    self.axpy(a=weights[m], x=fK[m], y=uNum[i+1])
            else:
                self.stepUpdate(uNum[i], uNodes[0], fEvals[0], out=uNum[i+1])

        return uNum



class ForwardEuler(GenericMultiNode):

    def evalPhi(self, uVals, fEvals, out, t0=0):
        m = len(uVals) - 1
        assert m > 0
        assert len(fEvals) in [m, m+1]

        tau = [t0] + (t0 + self.dt*self.nodes).tolist()

        # dt1 f0 + dt2 f1 + ... + dtm f{m-1}
        np.copyto(out, fEvals[0])
        out *= tau[1]-tau[0]
        for i in range(1, m):
            self.axpy(a=tau[i+1]-tau[i], x=fEvals[i], y=out)


    def nodeSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
        self.evalPhi([*uPrev, out], fEvals, out, t0=t0)
        out += rhs



class BackwardEuler(GenericMultiNode):

    def evalPhi(self, uVals, fEvals, out, t0=0):
        m = len(uVals) - 1
        assert m > 0
        assert len(fEvals) in [m, m+1]

        tau = [t0] + (t0 + self.dt*self.nodes).tolist()

        # dt1 f1 + dt2 f2 + ... + dtm f{m}
        if len(fEvals) == m:
            self.evalF(uVals[m], tau[m], out=out)   # f{m} not given
        else:
            np.copyto(out, fEvals[-1])   # f{m} given, use its value
        out *= tau[m]-tau[m-1]
        for i in range(m-1):
            self.axpy(a=tau[i+1]-tau[i], x=fEvals[i+1], y=out)

    def nodeSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
        assert len(uPrev) == len(fEvals)
        m = len(uPrev)
        assert m > 0
        tau = [t0] + (t0 + self.dt*self.nodes).tolist()

        rhs = np.zeros_like(out) + rhs
        for i in range(m-1):
            self.axpy(a=tau[i+1]-tau[i], x=fEvals[i+1], y=rhs)

        self.fSolve(tau[m]-tau[m-1], rhs, tau[m], out)
