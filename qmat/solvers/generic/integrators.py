#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specialized implementations of GenericMultiNode solvers
"""
import numpy as np

from qmat.solvers.generic import GenericMultiNode

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

    def phiSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
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

    def phiSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
        assert len(uPrev) == len(fEvals)
        m = len(uPrev)
        assert m > 0
        tau = [t0] + (t0 + self.dt*self.nodes).tolist()

        rhs = np.zeros_like(out) + rhs
        for i in range(m-1):
            self.axpy(a=tau[i+1]-tau[i], x=fEvals[i+1], y=rhs)

        self.fSolve(tau[m]-tau[m-1], rhs, tau[m], out)
