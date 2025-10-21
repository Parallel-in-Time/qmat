#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:26:04 2025

@author: cpf5546
"""
import numpy as np

import matplotlib.pyplot as plt
from time import time

from qmat import genQCoeffs, QDELTA_GENERATORS
from qmat.qcoeff.collocation import Collocation
from qmat.solvers.generic import LinearMultiNode

from qmat.solvers.generic.diffops import Dahlquist, Lorenz
from qmat.solvers.generic.integrators import ForwardEuler, BackwardEuler


pType = "Lorenz"
nPeriod = 1
nSteps = nPeriod*1000
tEnd = nPeriod*np.pi

corr = "FE"
useSDC = False
useWeights = False
nSweeps = 4


if pType == "Dahlquist":
    diffOp = Dahlquist()
elif pType == "Lorenz":
    diffOp = Lorenz()
nDOF = diffOp.u0.size

nodes, weights, Q = genQCoeffs(corr)
coll = Collocation(nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
gen = QDELTA_GENERATORS[corr](qGen=coll)
QDelta = gen.genCoeffs(k=[i+1 for i in range(nSweeps)])

prob = LinearMultiNode(diffOp, tEnd=tEnd, nSteps=nSteps)
Solver = BackwardEuler if corr == "BE" else ForwardEuler
if useSDC:
    solver = Solver(diffOp, nodes=coll.nodes, tEnd=tEnd, nSteps=nSteps)
else:
    regNodes = [0.25, 0.5, 0.75, 1]
    solver = Solver(diffOp, nodes=regNodes, tEnd=tEnd, nSteps=nSteps//4)

if useSDC:
    tBeg = time()
    uNumRef = prob.solveSDC(nSweeps, coll.Q, coll.weights if useWeights else None, QDelta)
else:
    tBeg = time()
    uNumRef = prob.solve(Q, weights)
tWall = time()-tBeg
tWall /= nSteps * nDOF
print(f"tWallScaled[linear] : {tWall:1.2e}s")

if useSDC:
    tBeg = time()
    uNum = solver.solveSDC(nSweeps, weights=useWeights)
else:
    tBeg = time()
    uNum = solver.solve()
tWall = time()-tBeg
tWall /= nSteps * nDOF
print(f"tWallScaled[generic] : {tWall:1.2e}s")
print(uNumRef[-1] - uNum[-1])

plt.figure(1)
plt.clf()
plt.plot(uNumRef[:, 0], uNumRef[:, 1], label="ref")
plt.plot(uNum[:, 0], uNum[:, 1], label="integrator")
plt.legend()
