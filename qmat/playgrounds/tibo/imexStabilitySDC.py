#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script investigating IMEX stability for SDC methods
"""
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

from qmat.qcoeff.collocation import Collocation
from qmat.qdelta import QDELTA_GENERATORS

from qmat.solvers.dahlquist import DahlquistIMEX


# Script parameters
nNodes = 4
nSweeps = 4
stepUpdate = False
implSweep = "MIN-SR-FLEX"
explSweep = "PIC"


# Script execution
lamE = np.linspace(0, 6, num=501)
ratio = np.logspace(-3, 2, num=301)
zI = -ratio[None, :]*lamE[:, None]
zE = 1j*lamE[:, None]

problem = DahlquistIMEX(zI, zE)

coll = Collocation(nNodes=nNodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT")

genI = QDELTA_GENERATORS[implSweep](qGen=coll)
genE = QDELTA_GENERATORS[explSweep](qGen=coll)

sweeps = [k+1 for k in range(nSweeps)]

uNum = problem.solveSDC(
    coll.Q, coll.weights if stepUpdate else None,
    genI.genCoeffs(k=sweeps), genE.genCoeffs(k=sweeps),
    nSweeps=nSweeps)

u1 = uNum[-1]
stab = np.abs(u1)
stab = np.clip(stab, 0, 1.2) # clip to ignore unstable area
error = np.abs(u1 - np.exp(zI+zE))


plt.figure("imex stability")
plt.clf()

coords = (ratio, lamE)
plt.contourf(*coords, stab, levels = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
plt.colorbar()
plt.contour(*coords, stab, levels=[1], colors="black")
plt.contour(*coords, error, levels=[1], colors="red", linestyles=":")
plt.contour(*coords, error, levels=[1e-1], colors="orange", linestyles="-.")
plt.contour(*coords, error, levels=[1e-2], colors="gray", linestyles="--")
plt.grid(True)
plt.xscale('log')
plt.ylabel(r"$\lambda_E \Delta t$")
plt.xlabel(r"advection $\quad\leftarrow\quad\frac{-\lambda_I}{\lambda_E}\quad\rightarrow\quad$ diffusion", fontsize=20)
plt.tight_layout()


def imStab(x):
    uSDC = DahlquistIMEX([0], [x*1j]).solveSDC(
        coll.Q, coll.weights if stepUpdate else None,
        genI.genCoeffs(k=sweeps), genE.genCoeffs(k=sweeps),
        nSweeps=nSweeps)
    return np.abs(uSDC[-1]) - 1

try:
    sol = sco.bisect(imStab, 1e-1, 1e2, xtol=1e-16)
    print(f"CFL max [SDC] : {sol}")
except:
    pass

plt.figure("stability on imaginary axis")
plt.clf()
plt.grid(True)
x = np.linspace(0, 6, num=501)
plt.plot(x, [imStab(s)[0] for s in x], label="RK")
plt.ylim(-1, 0.5)
plt.ylabel("over-amplification")
plt.xlabel(r"$\lambda_E \Delta t$")
plt.legend()
plt.tight_layout()

plt.show()
