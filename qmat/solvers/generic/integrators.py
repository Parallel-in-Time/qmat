#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specialized :class:`PhiSolver` classes implementing various time-integrators.
"""
import numpy as np

from qmat.solvers.generic import PhiSolver

class ForwardEuler(PhiSolver):
    r"""
    :math:`\phi`-based solver doing Forward Euler update between time nodes.

    It uses the following definition :

    .. math::

        \phi(u_0, u_1, ..., u_{m}, u_{m+1}) =
            \Delta\tau_{m+1} f(u_m, t_m) + ... + \Delta\tau_1 f(u_0, t_0),

    where :math:`\Delta\tau_{m} = t_{m+1} - t_{m}`.
    In particular, since it does not depends on the node solution
    :math:`u_{m+1}` (explicit scheme),
    its `phiSolve` method is replaced by an explicit evaluation of `evalPhi`.

    Parameters
    ----------
    diffOp : DiffOp
        Differential operator for the ODE.
    nodes : 1D array-like
        The time nodes :math:`\tau_1, ..., \tau_M`.
    tEnd : float, optional
        Final simulation time. The default is 1.
    nSteps : int, optional
        Number of simulation time-steps. The default is 1.
    t0 : float, optional
        Initial simulation time. The default is 0.
    """

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


class BackwardEuler(PhiSolver):
    r"""
    :math:`\phi`-based solver doing Backward Euler update between time nodes.

    It uses the following definition :

    .. math::

        \phi(u_0, u_1, ..., u_{m}, u_{m+1}) =
            \Delta\tau_{m+1} f(u_{m+1}, t_{m+1}) + ...
            + \Delta\tau_1 f(u_1, t_1),

    where :math:`\Delta\tau_{m} = t_{m+1} - t_{m}`.
    In particular, its `phiSolve` method is rewritten
    to depend directly on the `fSolve` method of the differential operator
    to avoid unecessary (re-)evaluations of :math:`f(u,t)`.

    Parameters
    ----------
    diffOp : DiffOp
        Differential operator for the ODE.
    nodes : 1D array-like
        The time nodes :math:`\tau_1, ..., \tau_M`.
    tEnd : float, optional
        Final simulation time. The default is 1.
    nSteps : int, optional
        Number of simulation time-steps. The default is 1.
    t0 : float, optional
        Initial simulation time. The default is 0.
    """

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
