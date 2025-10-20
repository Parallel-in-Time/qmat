#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:26:04 2025

@author: cpf5546
"""
import numpy as np
from scipy.linalg import blas

import matplotlib.pyplot as plt
from time import time

from qmat import genQCoeffs, QDELTA_GENERATORS
from qmat.qcoeff.collocation import Collocation
from qmat.solvers.generic import LinearMultiNode, ForwardEuler

pType = "Lorenz"

if pType == "Dahlquist":
    lam = 1j

    def evalF(u, t, out):
        out[0] = u[0]*lam.real - u[1]*lam.imag
        out[1] = u[1]*lam.real + u[0]*lam.imag

    u0 = np.array([1, 0], dtype=float)
    fSolve = None


elif pType == "Lorenz":
    sigma = 10
    rho = 28
    beta = 8/3

    def evalF(u, t, out):
        x, y, z = u
        out[0] = sigma*(y - x)
        out[1] = x*(rho - z) - y
        out[2] = x*y - beta*z

    u0 = np.array([5, -5, 20], dtype=float)

    newton = {
        "maxIter": 99,
        "tolerance": 1e-9,
        }

    gemv = blas.get_blas_funcs("gemv", dtype=u0.dtype)

    def fSolve(a, rhs, t, out):

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
            gemv(alpha=1.0, a=jacInv, x=res, beta=1.0, y=out, overwrite_y=True)

        fSolve = None   # because own implementation is cute, but still less efficient


nodes, weights, Q = genQCoeffs("FE")

coll = Collocation(nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
gen = QDELTA_GENERATORS["FE"](qGen=coll)
nSweeps = 2
QDelta = gen.genCoeffs(k=[i+1 for i in range(nSweeps)])


nSteps = 1000
tEnd = np.pi
prob = LinearMultiNode(u0, evalF, fSolve=fSolve, tEnd=tEnd, nSteps=nSteps)

solver = ForwardEuler(
    u0, evalF, nodes=[0.25, 0.5, 0.75, 1], fSolve=fSolve,
    tEnd=tEnd, nSteps=nSteps//4)

plt.figure(1)
plt.clf()

tBeg = time()

uNum = prob.solve(Q, weights)
# uNum = prob.solveSDC(coll.Q, None, QDelta, nSweeps=nSweeps)
plt.plot(uNum[:, 0], uNum[:, 1], label="ref")

uNum = solver.solve()
# uNum = solver.solveSDC(coll.Q, None, nSweeps=nSweeps)
plt.plot(uNum[:, 0], uNum[:, 1], label="integrator")

plt.legend()
tWall = time()-tBeg
tWall /= nSteps * np.size(u0)
print(f"tWallScaled : {tWall:1.2e}s")
