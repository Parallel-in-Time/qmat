#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and display orthogonal polynomials of any degree using `qmat`

.. literalinclude:: /../qmat/playgrounds/tibo/orthogonalPolynomials.py
   :language: python
   :linenos:
   :lines: 11-
"""
import numpy as np
import matplotlib.pyplot as plt
from qmat.nodes import NodesGenerator

deg = 100
"""polynomial degree"""

polyType = "CHEBY-1"
"""type of polynomial"""

t = np.linspace(-1, 1, num=1000000)
"""plotting points"""

gen = NodesGenerator(polyType)
alpha, beta = gen.getOrthogPolyCoefficients(deg+1)

# Generate monic polynomials (leading coefficient is 1)
if deg == 0:
    out = 0*t + 1.
else:
    pi = np.array([np.zeros_like(t) for i in range(3)])
    pi[1:] += 1
    for alpha_j, beta_j in zip(alpha, beta):
        pi[2] *= (t-alpha_j)
        pi[0] *= beta_j
        pi[2] -= pi[0]
        pi[0] = pi[1]
        pi[1] = pi[2]
    out = np.copy(pi[2])

# Scaling (depends on the kind of the polynomial)
if polyType == "CHEBY-1":
    out *= 2**deg
    ylim = (-1.1, 1.1)
elif polyType == ["CHEBY-2", "CHEBY-3", "CHEBY-4"]:
    out *= 2**(deg+(deg>0))
    ylim = (-1.6, 1.6)

plt.plot(t, out, label=f"{polyType}, $p={deg}$")
plt.ylim(*ylim)
plt.legend()
plt.xlabel("$t$")
plt.ylabel("$p(t)$")
plt.grid(True)
