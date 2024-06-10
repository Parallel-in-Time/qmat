#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for nodes generation
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

NODE_TYPES = ['EQUID', 'LEGENDRE', 'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4']
"""Available types (distributions) for nodes"""

QUAD_TYPES = ['GAUSS', 'RADAU-LEFT', 'RADAU-RIGHT', 'LOBATTO']
"""Available quadrature type for each node distribution"""

class NodesGenerator(object):
    """
    Class that can be used to generate generic distribution of nodes derived
    from Gauss quadrature rule.
    Its implementation is fully inspired from a `book of W. Gautschi
    <https://doi.org/10.1093/oso/9780198506720.001.0001>`_.

    Attributes
    ----------
    nodeType : str
        The type of node distribution
    quadType : str
        The quadrature type
    """
    def __init__(self, nodeType='LEGENDRE', quadType='LOBATTO'):
        """
        Parameters
        ----------
        nodeType : str, optional
            The type of node distribution, can be

            - EQUID : equidistant nodes
            - LEGENDRE : node distribution from Legendre polynomials
            - CHEBY-1 : node distribution from Chebychev polynomials (1st kind)
            - CHEBY-2 : node distribution from Chebychev polynomials (2nd kind)
            - CHEBY-3 : node distribution from Chebychev polynomials (3rd kind)
            - CHEBY-4 : node distribution from Chebychev polynomials (4th kind)

            The default is 'LEGENDRE'.

        quadType : str, optional
            The quadrature type, can be

            - GAUSS : inner point only, no node at boundary
            - RADAU-LEFT : only left boundary as node
            - RADAU-RIGHT : only right boundary as node
            - LOBATTO : left and right boundary as node

            The default is 'LOBATTO'.
        """

        # Check argument validity
        for arg, vals in zip(['nodeType', 'quadType'], [NODE_TYPES, QUAD_TYPES]):
            val = eval(arg)
            if val not in vals:
                raise ValueError(f"{arg}='{val}' not implemented, must be in {vals}")

        # Store attributes
        self.nodeType = nodeType
        self.quadType = quadType


    def getNodes(self, nNodes):
        """
        Computes a given number of quadrature nodes.

        Parameters
        ----------
        nNodes : int
            Number of nodes to compute.

        Returns
        -------
        nodes : np.1darray
            Nodes located in [-1, 1], in increasing order.
        """
        # Check number of nodes
        if self.quadType in ['LOBATTO', 'RADAU-LEFT'] and nNodes < 2:
            raise ValueError(
                f"nNodes must be larger than 2 for {self.quadType},"
                f" but got {nNodes}")
        elif nNodes < 1:
            raise ValueError("you surely want at least one node ;)")

        # Equidistant nodes
        if self.nodeType == 'EQUID':
            if self.quadType == 'GAUSS':
                return np.linspace(-1, 1, num=nNodes + 2)[1:-1]
            elif self.quadType == 'LOBATTO':
                return np.linspace(-1, 1, num=nNodes)
            elif self.quadType == 'RADAU-RIGHT':
                return np.linspace(-1, 1, num=nNodes + 1)[1:]
            elif self.quadType == 'RADAU-LEFT':
                return np.linspace(-1, 1, num=nNodes + 1)[:-1]

        # Quadrature nodes linked to orthogonal polynomials
        alpha, beta = self.getTridiagCoefficients(nNodes)
        nodes = eigh_tridiagonal(alpha, np.sqrt(beta[1:]))[0]
        nodes.sort()

        # Force overwrite boundary values if necessary (better precision)
        if self.quadType in ["LOBATTO", "RADAU-RIGHT"]:
            nodes[-1] = 1
        if self.quadType in ["LOBATTO", "RADAU-LEFT"]:
            nodes[0] = -1

        return nodes


    def getOrthogPolyCoefficients(self, num_coeff):
        """
        Produces a given number of analytic three-term recurrence coefficients.

        Parameters
        ----------
        num_coeff : int
            Number of coefficients to compute.

        Returns
        -------
        alpha : np.1darray
            The alpha coefficients of the three-term recurrence.
        beta : np.1darray
            The beta coefficients of the three-term recurrence.
        """
        if self.nodeType == 'LEGENDRE':
            k = np.arange(num_coeff, dtype=float)
            alpha = 0 * k
            beta = k**2 / (4 * k**2 - 1)
            beta[0] = 2
        elif self.nodeType == 'CHEBY-1':
            alpha = np.zeros(num_coeff)
            beta = np.full(num_coeff, 0.25)
            beta[0] = np.pi
            if num_coeff > 1:
                beta[1] = 0.5
        elif self.nodeType == 'CHEBY-2':
            alpha = np.zeros(num_coeff)
            beta = np.full(num_coeff, 0.25)
            beta[0] = np.pi / 2
        elif self.nodeType == 'CHEBY-3':
            alpha = np.zeros(num_coeff)
            alpha[0] = 0.5
            beta = np.full(num_coeff, 0.25)
            beta[0] = np.pi
        elif self.nodeType == 'CHEBY-4':
            alpha = np.zeros(num_coeff)
            alpha[0] = -0.5
            beta = np.full(num_coeff, 0.25)
            beta[0] = np.pi
        return alpha, beta


    def evalOrthogPoly(self, t, alpha, beta):
        """
        Evaluates the two higher order orthogonal polynomials corresponding
        to the given (alpha,beta) coefficients.

        Parameters
        ----------
        t : float or np.1darray
            The point where to evaluate the orthogonal polynomials.
        alpha : np.1darray
            The alpha coefficients of the three-term recurrence.
        beta : np.1darray
            The beta coefficients of the three-term recurrence.

        Returns
        -------
        pi[0] : float or np.1darray
            The second higher order orthogonal polynomial evaluation.
        pi[1] : float or np.1darray
            The higher oder orthogonal polynomial evaluation.
        """
        t = np.asarray(t, dtype=float)
        pi = np.array([np.zeros_like(t) for i in range(3)])
        pi[1:] += 1
        for alpha_j, beta_j in zip(alpha, beta):
            pi[2] *= t - alpha_j
            pi[0] *= beta_j
            pi[2] -= pi[0]
            pi[0] = pi[1]
            pi[1] = pi[2]
        return pi[0], pi[1]


    def getTridiagCoefficients(self, nNodes):
        """
        Computes recurrence coefficients for the tridiagonal Jacobian matrix,
        taking into account the quadrature type.

        Parameters
        ----------
        nNodes : int
            Number of nodes that should be computed from those coefficients.

        Returns
        -------
        alpha : np.1darray
            The modified alpha coefficients of the three-term recurrence.
        beta : np.1darray
            The modified beta coefficients of the three-term recurrence.
        """
        # Coefficients for Gauss quadrature type
        alpha, beta = self.getOrthogPolyCoefficients(nNodes)

        # If not Gauss quadrature type, modify the alpha/beta coefficients
        if self.quadType.startswith('RADAU'):
            b = -1.0 if self.quadType.endswith('LEFT') else 1.0
            b1, b2 = self.evalOrthogPoly(b, alpha[:-1], beta[:-1])[:2]
            alpha[-1] = b - beta[-1] * b1 / b2
        elif self.quadType == 'LOBATTO':
            a, b = -1.0, 1.0
            a2, a1 = self.evalOrthogPoly(a, alpha[:-1], beta[:-1])[:2]
            b2, b1 = self.evalOrthogPoly(b, alpha[:-1], beta[:-1])[:2]
            alpha[-1], beta[-1] = np.linalg.solve([[a1, a2], [b1, b2]], [a * a1, b * b1])
        return alpha, beta
