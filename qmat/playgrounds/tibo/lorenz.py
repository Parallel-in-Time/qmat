#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make use of the generic :class:`CoeffSolver` of `qmat` to solve the Lorenz equations

.. literalinclude:: /../qmat/playgrounds/tibo/lorenz.py
   :language: python
   :linenos:
   :lines: 11-
"""
import numpy as np
import matplotlib.pyplot as plt

from qmat import genQCoeffs, QDELTA_GENERATORS
from qmat.qcoeff.collocation import Collocation
from qmat.utils import Timer

from qmat.solvers.generic import CoeffSolver
from qmat.solvers.generic.diffops import Lorenz

tEnd = 10
nSteps = 1000
diffOp = Lorenz()
solver = CoeffSolver(diffOp, tEnd=tEnd, nSteps=nSteps)

nodes, weights, Q = genQCoeffs("RK4")
with Timer("RK solve", scale=nSteps, descr="tWall/step"):
    uRK = solver.solve(Q, weights)

coll = Collocation(nNodes=2, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
gen = QDELTA_GENERATORS["FE"](qGen=coll)
QDelta = gen.getQDelta()
with Timer("SDC solve", scale=nSteps, descr="tWall/step"):
    uSDC = solver.solveSDC(4, coll.Q, coll.weights, QDelta)

plt.figure("Solution")
times = np.linspace(0, tEnd, nSteps+1)
for i, v in enumerate(["x", "y", "z"]):
    p = plt.plot(times, uRK[:, i], label=f"{v} RK")
    plt.plot(times, uSDC[:, i], "--", c=p[0].get_color(), label=f"{v} SDC")
plt.legend()
plt.xlabel("time")
plt.ylabel("trajectory")
plt.gcf().set_size_inches(11, 6)
plt.tight_layout()
