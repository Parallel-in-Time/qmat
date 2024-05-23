#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule to generate Q matrices based on Collocation
"""
import numpy as np

from qmat.qcoeff import QGenerator, register
from qmat.nodes import NodesGenerator
from qmat.lagrange import LagrangeApproximation

@register
class Collocation(QGenerator):

    aliases = ["coll"]

    def __init__(self, nNodes, nodeType, quadType):

        # Generate nodes between [0, 1]
        nodes = NodesGenerator(nodeType, quadType).getNodes(nNodes)
        nodes += 1
        nodes /= 2
        np.round(nodes, 14, out=nodes)
        self._nodes = nodes

        # Lagrange approximation based on nodes
        approx = LagrangeApproximation(nodes)
        self._approx = approx

        # Compute Q (quadrature matrix) and weights
        Q = approx.getIntegrationMatrix([(0, tau) for tau in nodes])
        weights = approx.getIntegrationMatrix([(0, 1)]).ravel()
        self._Q, self._weights = Q, weights

    @property
    def nodes(self): return self._nodes

    @property
    def Q(self): return self._Q

    @property
    def weights(self): return self._weights

    @property
    def S(self):
        nodes = self._nodes
        pInts = [(0 if i == 0 else nodes[i-1], nodes[i])
                 for i in range(nodes.shape[0])]
        return self._approx.getIntegrationMatrix(pInts)

    @property
    def hCoeffs(self):
        return self._approx.getInterpolationMatrix([1]).ravel()
