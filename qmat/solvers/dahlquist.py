#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Solvers for the Dahlquist equation based on :math:`Q` coefficients,
also implementing SDC sweeps with given :math:`Q_\Delta` coefficients.
"""
import numpy as np


class Dahlquist():
    r"""
    Solver for the classical Dahlquist equation

    .. math::

        \frac{du}{dt} = \lambda u, \quad u(0)=u_0, \quad t \in [0,T].

    It can be used to solve the equation with multiple :math:`\lambda`
    values (multiple trajectories).

    Parameters
    ----------
    lam : scalar or array
        Value(s) used for :math:`\lambda`.
    u0 : scalar or array, optional
        Initial value :math:`\lambda`, must be compatible with `lam`.
        The default is 1.
    tEnd : float, optional
        Final simulation time :math:`T`. The default is 1.
    nSteps : float, optional
        Number of time-step to solve. The default is 1.
    """
    def __init__(self, lam, u0=1, tEnd=1, nSteps=1):
        self.u0 = u0
        """initial solution value"""

        self.tEnd = tEnd
        """final simulation time"""

        self.nSteps = nSteps
        """number of time-steps"""

        self.dt = tEnd/nSteps
        """time-step size"""

        self.lam = np.asarray(lam)
        r"""array storing the :math:`\lambda` values"""
        try:
            lamU = self.lam*u0
        except:
            raise ValueError("error when computing lam*u0")
        self.uShape = tuple(lamU.shape)
        """shape of the solution at a given time"""
        self.dtype = lamU.dtype
        """solution datatype"""


    @staticmethod
    def checkCoeff(Q, weights):
        """
        Check :math:`Q` coefficients and associated weights.

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like
            Quadrature weights associated to the nodes.

        Returns
        -------
        nNodes : int
            Number of nodes (stages).
        Q : np.2darray
            The :math:`Q` coefficients.
        weights : np.1darray
            Quadrature weights associated to the nodes.
        """
        Q = np.asarray(Q)
        nNodes = Q.shape[0]
        assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

        if weights is not None:
            weights = np.asarray(weights)
            assert weights.ndim == 1, \
                f"weights must be a 1D vector, not {weights}"
            assert weights.size == nNodes, \
                "weights size is not the same as the node size"
            assert np.allclose(weights.sum(), 1), \
                "weights sum must be equal to 1"
        else:
            assert np.allclose(Q.sum(axis=1)[-1], 1), \
                "last node must be 1 if weights are not given"

        return nNodes, Q, weights


    def solve(self, Q, weights):
        r"""
        Solve for all :math:`\lambda` using a direct solve of the :math:`Q`
        matrix, *i.e* for each time-step it solves :

        .. math::

            (I - \Delta{t}\lambda Q){\bf u} = {\bf u}_0,

        where :math:`{\bf u}_0` is the vector containing the initial solution
        of the time-step in each entry.
        The next step solution is computed using the **step update** :

        .. math::

            u_1 = u_0 + \Delta{t}\lambda{\bf w}^T{\bf u},

        or simply use the last **node solution** :math:`{\bf u}[-1]` if
        no weights are given (`weights=None`).

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like or None
            Quadrature weights associated to the nodes.
            If None, do not use them for the step update
            (requires last node equal to 1)

        Returns
        -------
        uNum : np.ndarray
            The solution at each time-steps (+ initial solution).
        """
        nNodes, Q, weights = self.checkCoeff(Q, weights)

        # Collocation problem matrix
        A = np.eye(nNodes) - self.lam[..., None, None]*self.dt*Q

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):
            b = np.ones(nNodes)*uNum[i][..., None]
            uNodes = np.linalg.solve(A, b[..., None])[..., 0]
            if weights is not None:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lam[..., None]*uNodes, weights)
            else:
                uNum[i+1] = uNodes[..., -1]

        return uNum


    @staticmethod
    def checkCoeffSDC(Q, weights, QDelta, nSweeps):
        r"""
        Check SDC coefficients

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like
            Quadrature weights associated to the nodes.
        QDelta : 2D or 3D array-like
            The :math:`Q_\Delta` coefficients (3D if changes with sweeps).
        nSweeps : int
            Number of sweeps.

        Returns
        -------
        nNodes : int
            Number of nodes.
        Q : np.2darray
            The :math:`Q` coefficients.
        weights : np.1darray
            Quadrature weights associated to the nodes.
        QDelta : np.2darray
            The :math:`Q_\Delta` coefficients for each sweep.
        nSweeps : int
            The number of sweeps.
        """
        Q = np.asarray(Q)
        nodes = Q.sum(axis=1)
        nNodes = nodes.size
        assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

        if weights is not None:
            weights = np.asarray(weights)
            assert weights.ndim == 1, "weights must be a 1D vector"
            assert weights.size == nNodes, \
                "weights size is not the same as the node size"
        else:
            assert np.allclose(nodes[-1], 1), \
                "last node must be 1 if weights are not given"

        QDelta = np.asarray(QDelta)
        if QDelta.ndim == 3:
            assert QDelta.shape == (nSweeps, nNodes, nNodes), \
                "inconsistent shape for QDelta"
        else:
            assert QDelta.shape == (nNodes, nNodes), \
                "inconsistent shape for QDelta"
            QDelta = np.repeat(QDelta[None, ...], nSweeps, axis=0)

        return nNodes, Q, weights, QDelta, nSweeps


    def solveSDC(self, Q, weights, QDelta, nSweeps):
        r"""
        Solve for all :math:`\lambda` using SDC sweeps, *i.e* solves for
        each time-step and sweep :math:`k` :

        .. math::

            (I - \Delta{t}\lambda Q_\Delta){\bf u}^{k+1}
            = {\bf u}_0 + \Delta{t}\lambda(Q - Q_\Delta){\bf u}^{k},

        where :math:`{\bf u}_0` is the vector containing the initial solution
        of the time-step in each entry and :math:`{\bf u}^0 = {\bf u}_0`
        (copy initialization).

        The next step solution is computed using the **step update** :

        .. math::

            u_1 = u_0 + \Delta{t}\lambda{\bf w}^T{\bf u}^{K},

        where :math:`K` is the total number of sweeps.
        If no weights are given (`weights=None`), it simply uses the last
        **node solution** :math:`{\bf u}[-1]`.

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like or None
            Quadrature weights associated to the nodes.
            If None, do not use them for the step update
            (requires last node equal to 1)
        QDelta : 2D or 3D array-like
            The :math:`Q_\Delta` coefficients (3D if changes with sweeps).
        nSweeps : int
            Number of sweeps.

        Returns
        -------
        uNum : np.ndarray
            The solution at each time-steps (+ initial solution).
        """
        nNodes, Q, weights, QDelta, nSweeps = self.checkCoeffSDC(
            Q, weights, QDelta, nSweeps)

        # Preconditioner for each sweeps
        P = np.eye(nNodes)[None, ...] \
            - self.lam[..., None, None, None]*self.dt*QDelta

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):

            uNodes = np.ones(nNodes)*uNum[i][..., None]
            uNodes = uNodes[..., :, None]   # shape [..., nNodes, 1]

            for k in range(nSweeps):

                b = uNum[i][..., None, None] \
                    + self.lam[..., None, None]*self.dt*(Q - QDelta[k]) @ uNodes

                # b has shape [..., nNodes, 1]
                # P[k] has shape [..., nNodes, nNodes]
                # output has shape [..., nNodes, 1]
                uNodes = np.linalg.solve(P[..., k, :, :], b)

            uNodes = uNodes[..., :, 0]  # back to shape [..., nNodes]

            if weights is None:
                uNum[i+1] = uNodes[..., -1]
            else:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lam[..., None]*uNodes, weights)

        return uNum


class DahlquistIMEX():
    r"""
    Solver for the IMEX Dahlquist equation

    .. math::

        \frac{du}{dt} = (\lambda_I + \lambda_E) u,
        \quad u(0)=u_0, \quad t \in [0,T].

    It can be used to solve the equation with multiple :math:`\lambda_I`
    and / or :math:`\lambda_E` values (multiple trajectories).

    Parameters
    ----------
    lamI : TYPE
        Value(s) used for :math:`\lambda_I`..
    lamE : scalar or array
        Value(s) used for :math:`\lambda_E`.
    u0 : scalar or array, optional
        Initial value :math:`\lambda`, must be compatible with `lam`.
        The default is 1.
    tEnd : float, optional
        Final simulation time :math:`T`. The default is 1.
    nSteps : float, optional
        Number of time-step to solve. The default is 1.
    """
    def __init__(self, lamI, lamE, u0=1, tEnd=1, nSteps=1):
        self.u0 = u0
        """initial solution value"""

        self.tEnd = tEnd
        """final simulation time"""

        self.nSteps = nSteps
        """number of time-steps"""

        self.dt = tEnd/nSteps
        """time-step size"""

        self.lamI = np.asarray(lamI)
        r"""array storing the :math:`\lambda_I` values"""
        self.lamE = np.asarray(lamE)
        r"""array storing the :math:`\lambda_E` values"""
        try:
            lamU = (self.lamI + self.lamE)*u0
        except:
            raise ValueError("error when computing (lamI + lamE)*u0")
        self.uShape = tuple(lamU.shape)
        """shape of the solution at one given time"""
        self.dtype = lamU.dtype
        """datatype of the solution array"""


    @staticmethod
    def checkCoeff(QI, wI, QE, wE):
        r"""
        Check IMEX :math:`Q` coefficients and assert their consistency.

        Parameters
        ----------
        QI : 2D array-like
            :math:`Q` coefficients used for :math:`\lambda_I`.
        wI : 1D array-like or None
            Weights used for the step update on :math:`\lambda_I`.
            If None, then step update is not done.
        QE : 2D array-like
            :math:`Q` coefficients used for :math:`\lambda_E`.
        wE : 1D array-like or None
            Weights used for the step update on :math:`\lambda_E`.
            If None, then step update is not done.

        Returns
        -------
        nNodes : int
            Number of nodes.
        QI : np.2darray
            :math:`Q` coefficients used for :math:`\lambda_I`.
        wI : np.1darray or None
            Weights used for the step update on :math:`\lambda_I`.
        QE : np.2darray
            :math:`Q` coefficients used for :math:`\lambda_E`.
        wE : np.1darray or None
            Weights used for the step update on :math:`\lambda_E`.
        useWeights : boll
            Wether or not the step update (using weights) is done.
        """
        QI, QE = np.asarray(QI), np.asarray(QE)
        assert np.allclose(QI.sum(axis=1), QE.sum(axis=1)), \
            "QI and QE do not correspond to the same nodes"

        nNodes = QI.shape[0]
        assert QI.shape == (nNodes, nNodes), "QI is not a square matrix"
        assert QI.shape == QE.shape, "QI and QE do not have the same shape"

        useWeights = True
        if wI is None or wE is None:
            assert wE is None and wI is None, \
                "it's either weights for everyone or no weight"
            useWeights = False

        return nNodes, QI, wI, QE, wE, useWeights


    def solve(self, QI, wI, QE, wE):
        r"""
        Solve for all :math:`\lambda_I` and :math:`\lambda_E`
        using a direct solve of the :math:`Q^I` and :math:`Q^E` matrices,
        *i.e* for each time-step it solves :

        .. math::

            (I - \lambda_I Q^I - \lambda_E Q^E){\bf u} = {\bf u}_0

        where :math:`{\bf u}_0` is the vector containing the initial solution
        of the time-step in each entry.
        The next step solution is computed using the IMEX **step update** :

        .. math::

            u_1 = u_0 + \Delta{t}\lambda_I{\bf w}_I^T{\bf u}
            + \Delta{t}\lambda_E{\bf w}_E^T{\bf u},

        or simply use the last **node solution** :math:`{\bf u}[-1]` if
        no weights are given (`wI=wE=None`).

        Parameters
        ----------
        QI : 2D array-like
            :math:`Q^I` coefficients used for :math:`\lambda_I`.
        wI : 1D array-like or None
            Weights used for the step update on :math:`\lambda_I`.
            If None, then step update is not done.
        QE : 2D array-like
            :math:`Q^E` coefficients used for :math:`\lambda_E`.
        wE : 1D array-like or None
            Weights used for the step update on :math:`\lambda_E`.
            If None, then step update is not done.

        Returns
        -------
        uNum : np.ndarray
            The solution at each time-steps (+ initial solution).
        """
        nNodes, QI, wI, QE, wE, useWeights = self.checkCoeff(QI, wI, QE, wE)

        # Collocation problem matrix
        A = np.eye(nNodes) \
            - self.lamI[..., None, None]*self.dt*QI \
            - self.lamE[..., None, None]*self.dt*QE

        # Solution vector for each time-step
        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
        uNum[0] = self.u0

        # Time-stepping loop
        for i in range(self.nSteps):

            b = np.ones(nNodes)*uNum[i][..., None]
            uNodes = np.linalg.solve(A, b[..., None])[..., 0]

            if useWeights:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(self.lamI[..., None]*uNodes, wI)
                uNum[i+1] += self.dt*np.dot(self.lamE[..., None]*uNodes, wE)
            else:
                uNum[i+1] = uNodes[..., -1]

        return uNum


    @staticmethod
    def checkCoeffSDC(Q, weights, QDeltaI, QDeltaE, nSweeps):
        r"""
        Check coefficients given for a IMEX SDC sweeps

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like or none
            Quadrature weights associated to the nodes. If None, last node is
            used for the step update.
        QDeltaE : 2D or 3D array-like
            The :math:`Q_\Delta^I` coefficients used for the :math:`\lambda_I`
            term (3D if changes with sweeps).
        QDeltaE : 2D or 3D array-like
            The :math:`Q_\Delta^E` coefficients used for the :math:`\lambda_E`
            term (3D if changes with sweeps).
        nSweeps : int
            Number of sweeps.

        Returns
        -------
        nNodes : int
            Number of nodes.
        Q : np.2darray
            The :math:`Q` coefficients.
        weights : np.1darray
            Quadrature weights associated to the nodes.
        QDeltaI : np.3darray
            The :math:`Q_\Delta^I` coefficients used for the :math:`\lambda_I`
            term for each sweeps.
        QDeltaE : np.3darray
            The :math:`Q_\Delta^E` coefficients used for the :math:`\lambda_E`
            term for each sweeps.
        nSweeps : int
            Number of SDC sweeps.
        """
        Q = np.asarray(Q)
        nodes = Q.sum(axis=1)
        nNodes = nodes.size
        assert Q.shape == (nNodes, nNodes), "Q is not a square matrix"

        if weights is not None:
            weights = np.asarray(weights)
            assert weights.ndim == 1, "weights must be a 1D vector"
            assert weights.size == nNodes, \
                "weights size is not the same as the node size"

        QDeltaI = np.asarray(QDeltaI)
        QDeltaE = np.asarray(QDeltaE)
        if QDeltaI.ndim == 3:
            assert QDeltaI.shape == (nSweeps, nNodes, nNodes), \
                "inconsistent shape for QDeltaI"
        else:
            assert QDeltaI.shape == (nNodes, nNodes), \
                "inconsistent shape for QDeltaE"
            QDeltaI = np.repeat(QDeltaI[None, ...], nSweeps, axis=0)
        if QDeltaE.ndim == 3:
            assert QDeltaE.shape == (nSweeps, nNodes, nNodes), \
                "inconsistent shape for QDeltaE"
        else:
            assert QDeltaE.shape == (nNodes, nNodes), \
                "inconsistent shape for QDeltaE"
            QDeltaE = np.repeat(QDeltaE[None, ...], nSweeps, axis=0)

        return nNodes, Q, weights, QDeltaI, QDeltaE, nSweeps


    def solveSDC(self, Q, weights, QDeltaI, QDeltaE, nSweeps):
        r"""
        Solve for all :math:`\lambda_I` and :math:`\lambda_E` using SDC sweeps,
        *i.e* for each time-step and sweep :math:`k` it solves :

        .. math::

            (I - \Delta{t}\lambda_I Q_\Delta^I - \Delta{t}\lambda_E Q_\Delta^I){\bf u}^{k+1}
            = {\bf u}_0 + \Delta{t}\left[
                \lambda Q - \lambda_I Q_\Delta^I - \lambda_E Q_\Delta^E\right]
            {\bf u}^{k},

        where :math:`{\bf u}_0` is the vector containing the initial solution
        of the time-step in each entry and :math:`{\bf u}^0 = {\bf u}_0`
        (copy initialization).
        The next step solution is computed using the **step update** :

        .. math::

            u_1 = u_0 + \Delta{t}\lambda{\bf w}^T{\bf u}^{K},

        where :math:`K` is the total number of sweeps.
        If no weights are given (`weights=None`), it simply uses the last
        **node solution** :math:`{\bf u}[-1]`.

        Parameters
        ----------
        Q : 2D array-like
            The :math:`Q` coefficients.
        weights : 1D array-like or none
            Quadrature weights associated to the nodes. If None, last node is
            used for the step update.
        QDeltaE : 2D or 3D array-like
            The :math:`Q_\Delta^I` coefficients used for the :math:`\lambda_I`
            term (3D if changes with sweeps).
        QDeltaE : 2D or 3D array-like
            The :math:`Q_\Delta^E` coefficients used for the :math:`\lambda_E`
            term (3D if changes with sweeps).
        nSweeps : int
            Number of sweeps.

        Returns
        -------
        uNum : np.ndarray
            The solution at each time-steps (+ initial solution).
        """
        nNodes, Q, weights, QDeltaI, QDeltaE, nSweeps = self.checkCoeffSDC(
            Q, weights, QDeltaI, QDeltaE, nSweeps)

        # Preconditioner for each sweeps
        P = np.eye(nNodes)[None, ...] \
            - self.lamI[..., None, None, None]*self.dt*QDeltaI \
            - self.lamE[..., None, None, None]*self.dt*QDeltaE

        uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
        uNum[0] = self.u0

        for i in range(self.nSteps):

            uNodes = np.ones(nNodes)*uNum[i][..., None]
            uNodes = uNodes[..., :, None]   # shape [..., nNodes, 1]

            for k in range(nSweeps):

                b = uNum[i][..., None, None] \
                    + self.lamI[..., None, None]*self.dt*(Q - QDeltaI[k]) @ uNodes \
                    + self.lamE[..., None, None]*self.dt*(Q - QDeltaE[k]) @ uNodes

                # b has shape [..., nNodes, 1]
                # P[k] has shape [..., nNodes, nNodes]
                # output has shape [..., nNodes, 1]
                uNodes = np.linalg.solve(P[..., k, :, :], b)

            uNodes = uNodes[..., :, 0]  # back to shape [..., nNodes]

            if weights is None:
                uNum[i+1] = uNodes[..., -1]
            else:
                uNum[i+1] = uNum[i]
                uNum[i+1] += self.dt*np.dot(
                    (self.lamI[..., None] + self.lamE[..., None])*uNodes, weights)

        return uNum
