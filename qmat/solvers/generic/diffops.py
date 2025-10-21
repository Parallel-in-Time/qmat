#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 17:00:11 2025

@author: cpf5546
"""
import numpy as np
from scipy.linalg import blas

from qmat.solvers.generic import DiffOperator


class Dahlquist(DiffOperator):

    def __init__(self, lam=1j):
        self.lam = lam
        u0 = np.array([1, 0], dtype=float)
        super().__init__(u0)


    def evalF(self, u, t, out):
        lam = self.lam
        out[0] = u[0]*lam.real - u[1]*lam.imag
        out[1] = u[1]*lam.real + u[0]*lam.imag


class Lorenz(DiffOperator):

    def __init__(self, sigma=10, rho=28, beta=8/3, nativeFSolve=False):
        self.params = [sigma, rho, beta]
        self.newton = {
            "maxIter": 99,
            "tolerance": 1e-9,
            }
        u0 = np.array([5, -5, 20], dtype=float)
        self.gemv = blas.get_blas_funcs("gemv", dtype=u0.dtype)
        super().__init__(u0)

        if nativeFSolve:
            self.fSolve = self.fSolve_NATIVE

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

        for n in range(newton["maxIter"]):
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
