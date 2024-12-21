#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule to generate Q matrices based on Collocation

Examples
--------

>>> from qmat.qcoeff.collocation import Collocation
>>> coll = Collocation(nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
>>> nodes, weights, Q = coll.genCoeffs()
>>> S = coll.S
"""
import numpy as np

from qmat.qcoeff import QGenerator, register
from qmat.nodes import NodesGenerator
from qmat.lagrange import LagrangeApproximation

@register
class Collocation(QGenerator):
    """
    Base class to generate :math:`Q`-coefficients for a Collocation method.

    Parameters
    ----------
    nNodes : int
        Number of collocation nodes.
    nodeType : str
        Type of node distributions, see :class:`qmat.nodes` for possible choices.
    quadType : str
        Quadrature type, see :class:`qmat.nodes` for possible choices.
    tLeft : float, optional
        Left boundary for the nodes. The default is 0.
    tRight : float, optional
        Right boundary for the nodes. The default is 1.
    """
    aliases = ["coll"]

    DEFAULT_PARAMS = {
        "nNodes": 4,
        "nodeType": "LEGENDRE",
        "quadType": "RADAU-RIGHT",
        }
    """Defaults parameters for getInstance"""

    def __init__(self, nNodes, nodeType, quadType, tLeft=0, tRight=1):
        self.nodeType, self.quadType = nodeType, quadType

        # Generate nodes between [0, 1]
        nodes = NodesGenerator(nodeType, quadType).getNodes(nNodes)
        nodes += 1
        nodes /= 2
        # Scale to [tLeft, tRight]
        nodes *= (tRight-tLeft)
        nodes += tLeft
        self.tLeft, self.tRight = tLeft, tRight
        # Safety when bound should be included ...
        if np.allclose(tLeft, nodes[0]): nodes[0] = tLeft
        if np.allclose(tRight, nodes[-1]): nodes[-1] = tRight
        self._nodes = nodes

        # Lagrange approximation based on nodes
        approx = LagrangeApproximation(nodes)
        self.approx = approx

        # Compute Q (quadrature matrix) and weights
        Q = approx.getIntegrationMatrix([(tLeft, tau) for tau in nodes])
        weights = approx.getIntegrationMatrix([(tLeft, tRight)]).ravel()
        self._Q, self._weights = Q, weights

        # For convergence tests
        if nodes.size == 3 and nodeType in ["CHEBY-3", "CHEBY-4"]:
            self.CONV_TEST_NSTEPS = [32, 64, 128]  # high error constant

    @property
    def nodes(self)->np.ndarray: return self._nodes

    @property
    def Q(self)->np.ndarray: return self._Q

    @property
    def weights(self)->np.ndarray: return self._weights

    @property
    def S(self)->np.ndarray:
        nodes = self._nodes
        pInts = [(self.tLeft if i == 0 else nodes[i-1], nodes[i])
                 for i in range(nodes.shape[0])]
        return self.approx.getIntegrationMatrix(pInts)

    @property
    def hCoeffs(self)->np.ndarray:
        return self.approx.getInterpolationMatrix([1]).ravel()

    @property
    def order(self)->int:
        M, nodeType, quadType = self.nodes.size, self.nodeType, self.quadType
        if nodeType != "LEGENDRE":
            if quadType in ["GAUSS", "LOBATTO"] \
                and nodeType in ["EQUID", "CHEBY-1", "CHEBY-2"]:
                return M + (M % 2)  # why ? the node symmetry I guess ...
            return M
        else:
            quadType = self.quadType
            if quadType == "GAUSS":
                return 2*M
            elif quadType.startswith("RADAU"):
                return 2*M-1
            elif quadType == "LOBATTO":
                return 2*M-2
