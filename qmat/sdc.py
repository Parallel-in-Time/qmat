#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to run SDC and evaluate its numerical error on simple problems.
"""
import numpy as np


def solveDahlquistSDC(lam, u0, T, nSteps, nSweeps, Q, QDelta,
                      weights=None):
    r"""
    Solve the Dahlquist problem with SDC.

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
    nSweeps : int
        Number of SDC sweeps.
    Q : np.ndarray
        Quadrature matrix :math:`Q` used for SDC.
    QDelta : np.ndarray
        Approximate quadrature matrix :math:`Q_\Delta` used for SDC.
    weights : np.ndarray, optional
        Quadrature weights to use for the prologation.
        If None, prolongation is not performed. The default is None.

    Returns
    -------
    uNum : np.ndarray
        Array containing the `nSteps+1` solutions :math:`\{u(0), ..., u(T)\}`.
    """
    uNum = np.zeros(nSteps+1, dtype=complex)
    uNum[0] = u0

    nNodes = Q.shape[0]

    dt = T/nSteps
    P = np.eye(nNodes) - lam*dt*QDelta
    for i in range(nSteps):

        uNodes = np.ones(nNodes)*uNum[i]

        for k in range(nSweeps):
            b = uNum[i] + lam*dt*(Q-QDelta) @ uNodes
            uNodes = np.linalg.solve(P, b)

        if weights is None:
            uNum[i+1] = uNodes[-1]
        else:
            uNum[i+1] = uNum[i] + lam*dt*weights.dot(uNodes)

    return uNum


def errorDahlquistSDC(lam, u0, T, nSteps, nSweeps, Q, QDelta,
                      weights=None, uNum=None):
    r"""
    Compute the time :math:`L_\infty` error of SDC.

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
    nSweeps : int
        Number of SDC sweeps.
    Q : np.ndarray
        Quadrature matrix :math:`Q` used for SDC.
    QDelta : np.ndarray
        Approximate quadrature matrix :math:`Q_\Delta` used for SDC.
    weights : np.ndarray, optional
        Quadrature weights to use for the prologation.
        If None, prolongation is not performed. The default is None.
    uNum : np.ndarray, optional
        Numerical solution, if not provided use the `solveDahlquist` method
        to compute the solution. The default is None.

    Returns
    -------
    float
        The :math:`L_\infty` norm.
    """
    if uNum is None:
        uNum = solveDahlquistSDC(
            lam, u0, T, nSteps, nSweeps, Q, QDelta,
            weights=weights)

    times = np.linspace(0, T, nSteps+1)
    uExact = u0 * np.exp(lam*times)
    return np.linalg.norm(uNum-uExact, ord=np.inf)


def getOrderSDC(coll, nSweeps, qDelta, prolongation):
    r"""
    Give the expected order of SDC after a fixed number of iterations.

    Parameters
    ----------
    coll : :class:`qmat.qcoeff.collocation.Collocation`
        The underlying `Collocation` class.
    nSweeps : int
        Number of sweeps for SDC.
    qDelta : str
        Type of the :math:`Q_\Delta` approximation used.
    prolongation : bool
        Wether or not the prolongation is done at the end.

    Returns
    -------
    order : int
        Expected order of the SDC time-integration.
    """
    # TODO : extend with additional results from
    # https://gitlab.inria.fr/sweet/sweet/-/blob/main/mule_local/python/sdc/qmatrix.py#L596

    nNodes, nodeType, quadType = coll.nodes.size, coll.nodeType, coll.quadType

    # Maximum order from the collocation problem
    maxOrder = coll.order

    # Order of SDC
    order = 0
    if nSweeps > 0:
        # first sweep
        if qDelta in ['TRAPAR', 'TRAP']:
            order += 2
        else:
            order += 1
        # rest of sweeps
        order += nSweeps-1
    # take into account prolongation
    if prolongation == "QUADRATURE":
        order += 1

    order = min(maxOrder, order)

    # Edge cases with bonus order (or malus some times ...)
    # TODO: couple with the Butcher theory from Joscha to retrieve this theoretically ...
    if prolongation == "QUADRATURE":  # COPY initialization
        if qDelta == "TRAP":
            if nSweeps == 1 and nNodes == 3 and nodeType == "EQUID" and quadType == "RADAU-LEFT":
                order += 1
            if nSweeps == 2 and nNodes == 4 and nodeType == "CHEBY-2" and quadType == "LOBATTO":
                order += 1
            if nSweeps == 2 and nNodes == 4 and nodeType == "CHEBY-3" and quadType == "RADAU-LEFT":
                order += 1
            if nSweeps == 2 and nNodes == 4 and nodeType == "CHEBY-4" and quadType in ["RADAU-RIGHT", "LOBATTO"]:
                order += 1
        if qDelta == "TRAPAR":
            if nSweeps == 2 and nNodes == 3 and nodeType == "CHEBY-3" and quadType == "RADAU-LEFT":
                order += 1
            if nSweeps == 2 and nNodes == 3 and nodeType == "CHEBY-4" and quadType == "RADAU-RIGHT":
                order += 1
            if nSweeps == 3 and nNodes == 3 and nodeType == "CHEBY-3" and quadType == "RADAU-LEFT":
                order += 1
        if qDelta == "BE":
            if nSweeps == 3 and nNodes == 3 and nodeType == "CHEBY-3" and quadType == "RADAU-LEFT":
                order += 1
            if nSweeps == 3 and nNodes == 3 and nodeType == "CHEBY-4" and quadType == "RADAU-RIGHT":
                order += 1
        if qDelta == "FE":
            if nSweeps == 3 and nNodes == 3 and nodeType == "CHEBY-1" and quadType == "RADAU-RIGHT":
                order += 1
            if nSweeps == 3 and nNodes == 3 and nodeType == "CHEBY-3" and quadType in ["RADAU-LEFT", "LOBATTO"]:
                order += 1
            if nSweeps == 3 and nNodes == 3 and nodeType == "CHEBY-4" and quadType == "RADAU-RIGHT":
                order += 1
        if qDelta == "BEPAR":
            if nSweeps == 3 and nNodes == 3 and nodeType == "EQUID" and quadType == "RADAU-LEFT":
                order += 1
            if nSweeps == 3 and nNodes == 3 and nodeType in ["CHEBY-1", "CHEBY-2"] and quadType in ["RADAU-LEFT", "RADAU-RIGHT"]:
                order += 1
            if nSweeps == 3 and nNodes == 3 and nodeType in ["CHEBY-3", "CHEBY-4"]:
                order += 1
        if qDelta == "TRAPAR":
            if nSweeps == 3 and nNodes == 4 and nodeType == "EQUID":
                order += 1
            if nSweeps == 3 and nNodes == 4 and nodeType in ["CHEBY-1", "CHEBY-2", "CHEBY-3", "CHEBY-4"]:
                order += 1

    if prolongation == "LASTNODE":
        if qDelta == "BE":
            if nSweeps == 4 and nNodes == 3 and nodeType == "CHEBY-4" and quadType == "RADAU-RIGHT":
                order += 1
        if qDelta == "FE":
            if nSweeps == 3 and nNodes == 3 and nodeType == "EQUID" and quadType == "RADAU-RIGHT":
                order += 1
            if nSweeps == 4 and nNodes == 3 and nodeType == "CHEBY-4" and quadType == "RADAU-RIGHT":
                order += 1
        if qDelta == "TRAP":
            if nSweeps == 2 and nNodes == 3 and nodeType in ["EQUID", "CHEBY-1", "CHEBY-2"] and quadType == "LOBATTO":
                order += 1
            if nSweeps == 2 and nNodes == 3 and nodeType == "LEGENDRE":
                order += 1
            if nSweeps == 2 and nNodes == 4 and nodeType == "EQUID" and quadType == "LOBATTO":
                order += 2
            if nSweeps == 2 and nNodes == 4 and nodeType == "EQUID" and quadType == "RADAU-RIGHT":
                order += 1
            if nSweeps == 2 and nNodes == 4 and nodeType in ["LEGENDRE", "CHEBY-1", "CHEBY-2", "CHEBY-3", "CHEBY-4"]:
                order += 1
            if nSweeps == 3 and nNodes == 3 and nodeType == "LEGENDRE" and quadType == "RADAU-RIGHT":
                order += 1
            if nSweeps == 4 and nNodes == 4 and nodeType == "LEGENDRE" and quadType == "LOBATTO":
                order += 1
            if nSweeps == 4 and nNodes == 4 and nodeType == "LEGENDRE" and quadType == "RADAU-RIGHT":
                order += 2
        if qDelta == "BEPAR":
            if nSweeps == 3 and nNodes == 2 and nodeType == "CHEBY-2" and quadType == "RADAU-RIGHT":
                order += 1
            if nSweeps == 4 and nNodes == 3 and nodeType in ["EQUID", "CHEBY-1", "CHEBY-2", "CHEBY-3", "CHEBY-4"] and quadType == "RADAU-RIGHT":
                order += 1
            if nSweeps == 4 and nNodes == 3 and nodeType in ["CHEBY-3", "CHEBY-4"] and quadType == "LOBATTO":
                order += 1
        if qDelta == "TRAPAR":
            if nSweeps == 3 and nNodes == 3 and nodeType == "CHEBY-4" and quadType == "RADAU-RIGHT":
                order += 1
            if nSweeps == 4 and nNodes == 4 and nodeType in ["EQUID", "CHEBY-1", "CHEBY-2", "CHEBY-3", "CHEBY-4"]:
                order += 1


    return order
