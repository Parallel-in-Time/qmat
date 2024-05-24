#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic example script
"""

from qmat import genQCoeffs, genQDeltaCoeffs


# Collocation method with user settings, BE sweep
nodes, weights, Q = genQCoeffs(
    "coll", nNodes=3, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
QDelta = genQDeltaCoeffs("MIN", Q=Q)

## Using the class directly
from qmat.qcoeff.collocation import Collocation

coll = Collocation(nNodes=4, nodeType="LEGENDRE", quadType="LOBATTO")
nodes, Q = coll.nodes, coll.Q

# Bucher coefficients
c, b, A = genQCoeffs("ERK4")
QDelta = genQDeltaCoeffs("Exact", Q=A)

from qmat.qcoeff.butcher import RK4

rk = RK4()
