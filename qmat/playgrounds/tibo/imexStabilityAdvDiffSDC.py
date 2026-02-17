#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script investigating IMEX stability for advection-diffusion solved with SDC
"""
import numpy as np
import matplotlib.pyplot as plt

from qmat.qcoeff.collocation import Collocation
from qmat.qdelta import QDELTA_GENERATORS

from qmat.solvers.dahlquist import DahlquistIMEX

# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
# -- SDC
nNodes = 4
nSweeps = 4
stepUpdate = False
implSweep = "LU"
explSweep = "FE"
nodeType="LEGENDRE"
quadType="RADAU-RIGHT"

# -- space discretization
dim = 3
advOrder = 4
diffOrder = 4

# -- plot options
cflAdvMax = 5
nSamplesCFL = 301
nWavenumbers = 16

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
ADV_COEFFS = {
    1: ([0, 1, -1], 1),
    2: ([1, 0, -1], 2),
    3: ([0, 2, 3, -6, 1], 6),
    4: ([-1, 8, 0, -8, 1], 12),
    5: ([0, -3, 30, 20, -60, 15, -2], 60),
    6: ([1, -9, 45, 0, -45, 9, -1], 60)
    }

DIFF_COEFFS = {
    2: ([1, -2, 1], 1),
    4: ([-1, 16, -30, 16, -1], 12),
    6: ([2, -27, 270, -490, 270, -27, 2], 180),
    }

if advOrder == np.inf:
    def zAdv(theta):
        return 1j*theta
else:
    def zAdv(theta):
        coeffs, div = ADV_COEFFS[advOrder]
        s = len(coeffs)
        exponents = range((s//2), -(s//2)-1, -1)
        symbol = sum(c*np.exp(1j*theta*e) for c, e in zip(coeffs, exponents))
        symbol /= div
        return symbol


if diffOrder == np.inf:
    def zDiff(theta):
        return -theta**2
else:
    def zDiff(theta):
        coeffs, div = DIFF_COEFFS[diffOrder]
        s = len(coeffs)
        exponents = range((s//2), -(s//2)-1, -1)
        symbol = sum(c*np.exp(1j*theta*e) for c, e in zip(coeffs, exponents))
        symbol /= div
        return symbol


theta = np.linspace(0, np.pi, num=nWavenumbers)
cflAdv = np.linspace(0, cflAdvMax, num=nSamplesCFL)
cflRatio = np.logspace(-2, 1, num=nSamplesCFL)

zE = -cflAdv[:, None, None]*zAdv(theta)[None, None, :]
zI = cflRatio[None, :, None]*cflAdv[:, None, None]*zDiff(theta)[None, None, :]
zI = (1+0j)*zI

zE *= dim
zI *= dim

print(f"Computing one SDC time-step on {zI.size} points ...")
problem = DahlquistIMEX(zI, zE)

coll = Collocation(nNodes=nNodes, nodeType=nodeType, quadType=quadType)

genI = QDELTA_GENERATORS[implSweep](qGen=coll)
genE = QDELTA_GENERATORS[explSweep](qGen=coll)

sweeps = [k+1 for k in range(nSweeps)]

uNum = problem.solveSDC(
    coll.Q, coll.weights if stepUpdate else None,
    genI.genCoeffs(k=sweeps), genE.genCoeffs(k=sweeps),
    nSweeps=nSweeps)
print(" -- done !")


u1 = uNum[-1]
stab = np.abs(u1).max(axis=-1)
stab = np.clip(stab, 0, 1.2) # clip to ignore unstable area
# error = np.abs(u1 - np.exp(zI+zE))


plt.figure("imex stability")
plt.clf()

coords = (cflRatio, cflAdv)
plt.contourf(*coords, stab, levels = [0.8, 1, 1.2], cmap="coolwarm")
plt.colorbar()
plt.contour(*coords, stab, levels=[1], colors="black")
# plt.contour(*coords, error, levels=[1], colors="red", linestyles=":")
# plt.contour(*coords, error, levels=[1e-1], colors="orange", linestyles="-.")
# plt.contour(*coords, error, levels=[1e-2], colors="gray", linestyles="--")
plt.grid(True)
plt.xscale('log')
plt.ylabel(r"$\sigma_{adv}$", fontsize=12)
plt.xlabel(r"$\frac{\sigma_{diff}}{\sigma_{adv}}$", fontsize=18)
plt.tight_layout()
plt.savefig("imexStabilityAdvDiff.pdf")
