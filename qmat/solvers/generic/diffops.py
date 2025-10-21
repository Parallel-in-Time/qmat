#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 17:00:11 2025

@author: cpf5546
"""
import numpy as np
from scipy.linalg import blas

from qmat.solvers.generic import DiffOp, registerDiffOp, DIFFOPS


@registerDiffOp
class Dahlquist(DiffOp):

    def __init__(self, lam=1j):
        self.lam = lam
        u0 = np.array([1, 0], dtype=float)
        super().__init__(u0)


    def evalF(self, u, t, out):
        lam = self.lam
        out[0] = u[0]*lam.real - u[1]*lam.imag
        out[1] = u[1]*lam.real + u[0]*lam.imag


@registerDiffOp
class Lorenz(DiffOp):
    r"""
    RHS of the Lorentz system, which can be written :

    .. math::
        \frac{dx}{dt} = \sigma (y-x), \; \frac{dy}{dt} = x (\rho - z) - y,
        \; \frac{dz}{dt} = xy - \beta z,

    with starting initial solution :math:`u_0=(x_0,y_0,z_0)=(5, -5, 20)`.
    Considering the three dimensional vector :math:`u=(x,y,z)`, the formal
    expression of :math:`f` is then

    .. math::
        f(u,t) = [ \sigma (y-x), x (\rho - z) - y, xy - \beta z ]

    Parameters
    ----------
    sigma: float, optional
        The :math:`\sigma` parameter (default=10).
    rho: float, optional
        The :math:`\rho` parameter (default=28).
    beta: float, optional
        The :math:`\beta` parameter (default=8/3).
    nativeFSolve: bool, optional
        Wether or not using the native fSolve method (default is False).
    """

    def __init__(self, sigma=10, rho=28, beta=8/3, nativeFSolve=False):
        self.params = [sigma, rho, beta]
        r"""list containing :math:`\sigma`, :math:`\rho` and :math:`\beta`"""

        self.newton = {
            "maxIter": 99,
            "tolerance": 1e-9,
            }
        """parameters for the Newton iteration used in native fSolve"""

        u0 = np.array([5, -5, 20], dtype=float)
        self.gemv = blas.get_blas_funcs("gemv", dtype=u0.dtype)
        """level-2 blas gemv function used in the native solver (just for flex, doesn't bring anything)"""

        super().__init__(u0)
        if nativeFSolve:
            self.fSolve = self.fSolve_NATIVE


    @classmethod
    def test(cls):
        super().test(instance=cls())
        super().test(instance=cls(nativeFSolve=True))


    def evalF(self, u, t, out):
        sigma, rho, beta = self.params
        x, y, z = u
        out[0] = sigma*(y - x)
        out[1] = x*(rho - z) - y
        out[2] = x*y - beta*z

    def fSolve_NATIVE(self, a, rhs, t, out):
        sigma, rho, beta = self.params
        newton = self.newton

        rhsX, rhsY, rhsZ = rhs
        a2 = a**2
        a3 = a**3

        for _ in range(newton["maxIter"]):
            x, y, z = out

            res = np.array([
                x - a*sigma*(y - x)     - rhsX,
                y - a*(x*(rho - z) - y) - rhsY,
                z - a*(x*y - beta*z)    - rhsZ,
            ])

            resNorm = np.linalg.norm(res, np.inf)
            if resNorm <= newton["tolerance"]:
                break
            if np.isnan(resNorm):
                break

            factor = -1.0 / (
                a3*sigma*(x*(x + y) + beta*(-rho + z + 1))
                + a2*(beta*sigma + beta - rho*sigma + sigma + x**2 + sigma*z)
                + a*(beta + sigma + 1) + 1
            )

            jacInv = factor * np.array([
                [
                    beta*a2 + a2*(x**2) + beta*a + a + 1,
                    beta*a2*sigma + a*sigma,
                    -a2*sigma*x,
                ],
                [
                    beta*a2*rho - a2*x*y - beta*a2*z + a*rho - a*z,
                    beta*a2*sigma + beta*a + a*sigma + 1,
                    -(a2*sigma + a)*x,
                ],
                [
                    a2*rho*x - a2*x*z + a2*y + a*y,
                    a2*sigma*x + a2*sigma*y + a*x,
                    -a2*rho*sigma + a2*sigma*(1 + z) + a*sigma + a + 1,
                ],
            ])

            # out += jacInv @ res
            self.gemv(alpha=1.0, a=jacInv, x=res, beta=1.0, y=out, overwrite_y=True)


@registerDiffOp
class ProtheroRobinson(DiffOp):
    r"""
    Implement the Prothero-Robinson problem:

    .. math::
        \frac{du}{dt} = -\frac{u-g(t)}{\epsilon} + \frac{dg}{dt}, \quad u(0) = g(0).,

    with :math:`\epsilon` a stiffness parameter, that makes the problem more stiff
    the smaller it is (usual taken value is :math:`\epsilon=1e^{-3}`).
    Exact solution is given by :math:`u(t)=g(t)`, and this implementation uses
    :math:`g(t)=\cos(t)`.

    Implement also the non-linear form of this problem:

    .. math::
        \frac{du}{dt} = -\frac{u^3-g(t)^3}{\epsilon} + \frac{dg}{dt}, \quad u(0) = g(0).

    To use an other exact solution, one just have to derivate this class
    and overload the `g` and `dg` methods. For instance,
    to use :math:`g(t)=e^{-0.2*t}`, define and use the following class:

    >>> class MyProtheroRobinson(ProtheroRobinson):
    >>>
    >>>     def g(self, t):
    >>>         return np.exp(-0.2 * t)
    >>>
    >>>     def dg(self, t):
    >>>         return (-0.2) * np.exp(-0.2 * t)

    Parameters
    ----------
    epsilon : float, optional
        Stiffness parameter. The default is 1e-3.
    nonLinear : bool, optional
        Wether or not to use the non-linear form of the problem. The default is False.
    nativeFSolve : bool, optional
        Wether or not use the native fSolver using exact Jacobian. The default is True.

    Reference
    ---------
    A. Prothero and A. Robinson, On the stability and accuracy of one-step methods for solving
    stiff systems of ordinary differential equations, Mathematics of Computation, 28 (1974),
    pp. 145–162.
    """

    def __init__(self, epsilon=1e-3, nonLinear=False, nativeFSolve=True):
        self.epsilon = epsilon
        self.newton = {
            "maxIter": 200,
            "tolerance": 5e-15,
            }
        self.evalF = self.evalF_NONLIN if nonLinear else self.evalF_LIN
        self.jac = self.jac_NONLIN if nonLinear else self.jac_LIN
        if nativeFSolve:
            self.fSolve = self.fSolve_NATIVE
        super().__init__([self.g(0)])

    @classmethod
    def test(cls):
        default = cls()
        assert not default.nonLinear, "default ProtheroRobinson DiffOp is not linear"
        super().test(instance=default)
        super().test(instance=cls(nativeFSolve=True))

    @property
    def nonLinear(self):
        return self.evalF == self.evalF_NONLIN

    # -------------------------------------------------------------------------
    # g function (analytical solution), and its first derivative
    # -------------------------------------------------------------------------
    def g(self, t):
        return np.cos(t)

    def dg(self, t):
        return -np.sin(t)

    # -------------------------------------------------------------------------
    # f(u,t) and Jacobian functions
    # -------------------------------------------------------------------------
    def evalF(self, u, t, out):
        raise NotImplementedError("evalF was not set on initialization")

    def evalF_LIN(self, u, t, out):
        np.copyto(out, -self.epsilon**(-1) * (u - self.g(t)) + self.dg(t))

    def evalF_NONLIN(self, u, t, out):
        np.copyto(out, -self.epsilon**(-1) * (u**3 - self.g(t)**3) + self.dg(t))

    def jac(self, u, t):
        raise NotImplementedError("jac was not set on initialization")

    def jac_LIN(self, u, t):
        return -self.epsilon**(-1)

    def jac_NONLIN(self, u, t):
        return -self.epsilon**(-1) * 3*u**2

    def fSolve_NATIVE(self, a, rhs, t, out):
        newton = self.newton
        u = out

        for _ in range(newton["maxIter"]):
            res = np.array([0.0])
            self.evalF(u, t, out=res)
            res *= -a
            res += u
            res -= rhs
            resNorm = np.linalg.norm(res, np.inf)
            if resNorm <= newton["tolerance"]:
                break
            if np.isnan(resNorm):
                break

            jac = 1 - a * self.jac(u, t)
            u -= res / jac

assert len(DIFFOPS) > 0, "something is wrong with DiffOp registration"
