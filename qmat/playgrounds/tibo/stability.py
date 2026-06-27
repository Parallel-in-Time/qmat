#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute accuracy and stability of a monolitic SDC scheme

.. literalinclude:: /../qmat/playgrounds/tibo/stability.py
   :language: python
   :linenos:
   :lines: 11-
"""
import numpy as np
import matplotlib.pyplot as plt

from qmat.qcoeff.collocation import Collocation
from qmat.qdelta import QDELTA_GENERATORS

from qmat.solvers.dahlquist import Dahlquist

# Script parameters
nNodes = 4
nSweeps = 1
sweepType = "SOE"

# Script execution
re = np.linspace(-4.5, 1, num=200)
im = np.linspace(0, 5.0, num=201)
lam = re[None, :] + 1j*im[:, None]

problem = Dahlquist(lam)
coll = Collocation(nNodes=nNodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
approx = QDELTA_GENERATORS[sweepType](qGen=coll)

sweeps = [k+1 for k in range(nSweeps)]

uNum = problem.solveSDC(
    coll.Q, None, approx.genCoeffs(k=sweeps), nSweeps=nSweeps)


u1 = uNum[-1]
stab = np.abs(u1)
stab = np.clip(stab, 0, 1.2) # clip to ignore unstable area
error = np.abs(u1 - np.exp(lam))

plt.figure(f"{sweepType}_M{nNodes}_K{nSweeps}")
coords = (re, im)
plt.contourf(*coords, stab, levels = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
plt.colorbar()
plt.contour(*coords, stab, levels=[1], colors="black")
plt.contour(*coords, error, levels=[1], colors="red", linestyles=":")
plt.contour(*coords, error, levels=[1e-1], colors="orange", linestyles="-.")
plt.contour(*coords, error, levels=[1e-2], colors="gray", linestyles="--")
plt.grid(True)
plt.ylabel(r"$Im(\lambda)$")
plt.xlabel(r"$Re(\lambda)$")
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
