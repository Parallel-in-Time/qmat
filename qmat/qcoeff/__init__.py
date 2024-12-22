#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the base abstract class to generate :math:`Q`-coefficients (Butcher tables) :
the :class:`QGenerator` 🚀

Each submodule contains specializations of this class for many kind of
methods :

- :class:`collocation` : Collocation based
- :class:`butcher` : Runge-Kutta based (Butcher tables)
"""
import numpy as np

from qmat.utils import checkOverriding, storeClass, importAll
from qmat.lagrange import LagrangeApproximation

class QGenerator(object):
    """Base abstract class for all :math:`Q`-coefficients generators"""

    @classmethod
    def getInstance(cls):
        """Provide an instance of this QGenerator using default parameters."""
        try:
            return cls()
        except TypeError:
            return cls(**cls.DEFAULT_PARAMS)

    @property
    def nodes(self):
        r"""Nodes :math:`\tau` (:math:`c` coefficients in Butcher table)"""
        raise NotImplementedError("mouahahah")

    @property
    def Q(self):
        r""":math:`Q` coefficients (:math:`A` Butcher table)"""
        raise NotImplementedError("mouahahah")

    @property
    def weights(self):
        r"""Weights :math:`\omega` (:math:`b` coefficients in Butcher table)"""
        raise NotImplementedError("mouahahah")

    @property
    def weightsEmbedded(self):
        """Weights for a secondary lower order method from the same stages."""
        raise NotImplementedError(f"no embedded weights implemented for {type(self).__name__}")

    @property
    def nNodes(self)->int:
        """Number of nodes (or stages) for this QGenerator"""
        return self.nodes.size

    @property
    def rightIsNode(self)->bool:
        """Wether or not the last nodes is the right boundary"""
        return self.nodes[-1] == 1.

    @property
    def T(self)->np.ndarray:
        """Transfer matrix from zero-to-nodes to node-to-node"""
        M = self.Q.shape[0]
        T = np.eye(M)
        T[1:,:-1][np.diag_indices(M-1)] = -1
        return T

    @property
    def S(self)->np.ndarray:
        """Quadrature matrix in node to node (N2N)"""
        Q = np.asarray(self.Q)
        M = self.Q.shape[0]
        T = np.eye(M)
        T[1:,:-1][np.diag_indices(M-1)] = -1
        return T @ Q

    @property
    def Tinv(self)->np.ndarray:
        """Transfer matrix from node-to-node to zero-to-node"""
        M = self.Q.shape[0]
        return np.tri(M)

    @property
    def hCoeffs(self)->np.ndarray:
        """:math:`h` interpolation coefficients for the right boundary"""
        approx = LagrangeApproximation(self.nodes)
        return approx.getInterpolationMatrix([1]).ravel()

    def genCoeffs(self, form="Z2N", hCoeffs=False, embedded=False):
        """
        Generate :math:`Q`-coefficients of this :class:`QGenerator` object.

        Parameters
        ----------
        form : str, optional
            Write coefficients in zero-to-nodes (Z2N) or node-to-node (N2N).
            The default is "Z2N".
        hCoeffs : bool, optional
            Wether or not returning the :math:`h` coefficients. The default is False.
        embedded : bool, optional
            Wether or not returning the embedded :math:`h` coefficients.
            The default is False.

        Returns
        -------
        out : tuple
            Contains (nodes, weights, Q).
            If `hCoeffs=True`, returns (nodes, weights, Q, hCoeffs).
            If `embedded=True`, `weights` is a 2xM array containing embedded weights in `weights[1]`.
        """
        if form == "Z2N":
            mat = self.Q
        elif form == "N2N":
            mat = self.S
        else:
            raise ValueError(f"form must be Z2N or N2N, not {form}")
        out = [self.nodes, self.weights, mat]
        if embedded:
            out[1] = np.vstack([out[1], self.weightsEmbedded])
        if hCoeffs:
            out.append(self.hCoeffs)
        return out

    @property
    def order(self):
        """Global convergence order of the method"""
        raise NotImplementedError("mouahahah")

    @property
    def orderEmbedded(self)->int:
        """Global convergence order of the associated embedded method"""
        return self.order - 1

    def solveDahlquist(self, lam, u0, T, nSteps, useEmbeddedWeights=False):
        r"""
        Solve the Dahlquist test problem

        .. math::

            \frac{du}{dt} = \lambda u, \quad t \in [0, T], \quad u(0)=u_0

        Parameters
        ----------
        lam : complex or float
            The :math:`\lambda` coefficient.
        u0 : complex or float
            The initial solution :math:`u_0`.
        T : float
            Final time :math:`T`.
        nSteps : int
            Number of time-step for the whole :math:`[0,T]` interval.
        useEmbeddedWeights : bool, optional
            Wether or not use the embedded weights for the prolongation. The default is False.

        Returns
        -------
        uNum : np.ndarray
            Array containing the `nSteps+1` solutions :math:`\{u(0), ..., u(T)\}`.
        """
        nodes, weights, Q = self.nodes, self.weights, self.Q

        if useEmbeddedWeights:
            weights = self.weightsEmbedded

        uNum = np.zeros(nSteps+1, dtype=complex)
        uNum[0] = u0

        dt = T/nSteps
        A = np.eye(nodes.size) - lam*dt*Q
        for i in range(nSteps):
            b = np.ones(nodes.size)*uNum[i]
            uStages = np.linalg.solve(A, b)
            uNum[i+1] = uNum[i] + lam*dt*weights.dot(uStages)

        return uNum

    def errorDahlquist(self, lam, u0, T, nSteps, uNum=None, useEmbeddedWeights=False):
        """
        Compute :math:`L_\infty` error in time for the Dahlquist problem

        Parameters
        ----------
        lam : complex or float
            The :math:`\lambda` coefficient.
        u0 : complex or float
            The initial solution :math:`u_0`.
        T : float
            Final time :math:`T`.
        nSteps : int
            Number of time-step for the whole :math:`[0,T]` interval.
        uNum : np.ndarray, optional
            Numerical solution, if not provided use the `solveDahlquist` method
            to compute the solution. The default is None.
        useEmbeddedWeights : bool, optional
            Wether or not use the embedded weights for the prolongation. The default is False.

        Returns
        -------
        float
            The :math:`L_\infty` norm.
        """
        if uNum is None:
            uNum = self.solveDahlquist(lam, u0, T, nSteps, useEmbeddedWeights=useEmbeddedWeights)
        times = np.linspace(0, T, nSteps+1)
        uExact = u0 * np.exp(lam*times)
        return np.linalg.norm(uNum-uExact, ord=np.inf)


Q_GENERATORS = {}
"""Dictionary containing all specialized :class:`QGenerator` classes, with all their aliases"""

def register(cls:QGenerator)->QGenerator:
    """Class decorator to register a specialized :class:`QGenerator` class in qmat"""
    # Check for correct overriding
    for name in ["nodes", "Q", "weights", "order"]:
        checkOverriding(cls, name)
    # Check that TEST_PARAMS are given and valid if no default constructor
    try:
        cls()
    except TypeError:
        try:
            params = cls.DEFAULT_PARAMS
        except AttributeError:
            raise AttributeError(
                f"{cls.__name__} requires DEFAULT_PARAMS attribute"
                " since it has no default constructor")
        try:
            cls(**params)
        except:
            raise TypeError(
                f"{cls.__name__} could not be instantiated with DEFAULT_PARAMS")
    # Store class (and aliases)
    storeClass(cls, Q_GENERATORS)
    return cls

def genQCoeffs(qType, form="Z2N", hCoeffs=False, embedded=False, **params):
    """
    Generate :math:`Q`-coefficients for a given method

    Parameters
    ----------
    qType : str
        Name (or alias) of the QGenerator.
    form : str, optional
        Write coefficients in zero-to-nodes (Z2N) or node-to-node (N2N).
        The default is "Z2N".
    hCoeffs : bool, optional
        Wether or not returning the :math:`h` coefficients. The default is False.
    embedded : bool, optional
        Wether or not returning the embedded :math:`h` coefficients.
        The default is False.
    **params :
        Parameters to be used to instantiate the QGenerator.

    Returns
    -------
    out : tuple
        Contains (nodes, weights, Q).
        If `hCoeffs=True`, returns (nodes, weights, Q, hCoeffs).
        If `embedded=True`, `weights` is a 2xM array containing embedded weights in `weights[1]`.
    """
    try:
        Generator = Q_GENERATORS[qType]
    except KeyError:
        raise ValueError(f"{qType=!r} is not available")
    gen = Generator(**params)
    return gen.genCoeffs(form, hCoeffs, embedded)


# Import all local submodules
__all__ = ["genQCoeffs", "QGenerator", "Q_GENERATORS", "register"]
importAll(locals(), __all__, __path__, __name__, __import__)
