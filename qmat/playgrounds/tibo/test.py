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
from qmat.solvers.generic import CoeffSolver

from qmat.solvers.generic.diffops import Dahlquist, Lorenz, ProtheroRobinson
from qmat.solvers.generic.integrators import ForwardEuler, BackwardEuler


pType = "Lorenz"
nPeriod = 1
nSteps = nPeriod*1000
tEnd = nPeriod*np.pi

corr = "BE"
useSDC = True
useWeights = True
nSweeps = 1

if pType == "Dahlquist":
    diffOp = Dahlquist()
elif pType == "Lorenz":
    diffOp = Lorenz()
elif pType == "ProtheroRobinson":
    nSteps *= 10
    diffOp = ProtheroRobinson(nonLinear=False)
nDOF = diffOp.u0.size

nodes, weights, Q = genQCoeffs(corr)
coll = Collocation(nNodes=2, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
gen = QDELTA_GENERATORS[corr](qGen=coll)
QDelta = gen.genCoeffs(k=[i+1 for i in range(nSweeps)])

prob = CoeffSolver(diffOp, tEnd=tEnd, nSteps=nSteps)
Solver = BackwardEuler if corr == "BE" else ForwardEuler
if useSDC:
    solver = Solver(diffOp, nodes=coll.nodes, tEnd=tEnd, nSteps=nSteps)
else:
    regNodes = [0.25, 0.5, 0.75, 1]
    solver = Solver(diffOp, nodes=regNodes, tEnd=tEnd, nSteps=nSteps//4)

if useSDC:
    tBeg = time()
    uRef = prob.solveSDC(nSweeps, coll.Q, coll.weights if useWeights else None, QDelta)
else:
    tBeg = time()
    uRef = prob.solve(Q, weights)
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
print(uRef[-1] - uNum[-1])

plt.figure(1)
plt.clf()
if pType == "ProtheroRobinson":
    times = np.linspace(0, tEnd, nSteps+1)
    plt.plot(times, uRef[:, 0], label="ref")
    if useSDC:
        plt.plot(times, uNum[:, 0], label="integrator")
    else:
        plt.plot(times[::4], uNum[:, 0], label="integrator")
else:
    plt.plot(uRef[:, 0], uRef[:, 1], label="ref")
    plt.plot(uNum[:, 0], uNum[:, 1], label="integrator")
plt.legend()
