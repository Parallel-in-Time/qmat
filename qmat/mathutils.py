#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for math utility functions
"""
import numpy as np


def numericalOrder(nSteps, err):
    """
    Compute numerical order from two vectors containing the error and the number of time-steps.

    Parameters
    ----------
    nSteps : np.1darray or list
        Different number of steps to compute the error.
    err : np.1darray
        Diffenrent error values associated to the number of steps.

    Returns
    -------
    beta : float
        Order coefficient computed through linear regression.
    rmse : float
        The root mean square error of the linear regression.
    """
    nSteps = np.asarray(nSteps)
    x, y = np.log10(1/nSteps), np.log10(err)

    # Compute regression coefficients and rmse
    xMean = x.mean()
    yMean = y.mean()
    sX = ((x-xMean)**2).sum()
    sXY = ((x-xMean)*(y-yMean)).sum()

    beta = sXY/sX
    alpha = yMean - beta*xMean

    yHat = alpha + beta*x
    rmse = ((y-yHat)**2).sum()**0.5
    rmse /= x.size**0.5

    return beta, rmse


def lduFactorization(A:np.ndarray):
    """
    Perform LDU factorization on a square matrix A.
    
    Parameters
    ----------
    A : np.2darray 
        The square matrix to factorize (n x n).
    
    Returns
    -------
    L : np.2darray
        Lower triangular matrix with ones on the diagonal.
    D : np.2darray 
        Diagonal matrix.
    U : np.2darray 
        Upper triangular matrix with ones on the diagonal.
    """
    # Ensure A is a square matrix
    n, m = A.shape
    assert n == m, "Matrix A must be square."

    # Initialize L, D, U matrices
    L = np.eye(n)
    D = np.zeros((n, n))
    U = np.eye(n)

    # Decompose A into L, D, U
    for i in range(n):
        # Compute D[i, i] as the diagonal element
        D[i, i] = A[i, i] - np.sum(L[i, :i] * D[:i, :i].diagonal() * U[:i, i])

        for j in range(i + 1, n):
            # Compute elements for L below the diagonal
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * D[:i, :i].diagonal() * U[:i, i])) / D[i, i]
            
            # Compute elements for U above the diagonal
            U[i, j] = (A[i, j] - np.sum(L[i, :i] * D[:i, :i].diagonal() * U[:i, j])) / D[i, i]

    return L, D, U


def getExtrapolationMatrix(nodes, times, pOrder=None):
    """
    Generate a polynomial based extrapolation matrix,
    base on polynomial regression of a function represented
    on given nodes.

    Parameters
    ----------
    nodes : np.1darray like, shape (M,)
        The nodes where function values are known.
    times : np.1darray like, shape (N,)
        The times where to extrapolate the polynomial.
    pOrder : int
        Order of the polynomial regression on the node values
        (default = len(nodes)-1).

    Returns
    -------
    P : np.2darray, shape (N, M)
        Extrapolation matrix, that can be used on any node values.
    """
    if pOrder is None: pOrder = np.size(nodes)-1
    X = np.vander(nodes, N=pOrder+1, increasing=True)
    T = np.vander(times, N=pOrder+1, increasing=True)
    P = T @ np.linalg.solve(X.T @ X, X.T)
    return P