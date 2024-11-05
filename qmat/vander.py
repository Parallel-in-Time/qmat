#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for Polynomial approximation using Vandermonde matrices
"""
import numpy as np

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
    Pe : np.2darray, shape (N, M)
        Extrapolation matrix, that can be used on any node values.
    """
    if pOrder is None: pOrder = np.size(nodes)-1
    X = np.vander(nodes, N=pOrder+1, increasing=True)
    T = np.vander(times, N=pOrder+1, increasing=True)
    Pe = T @ np.linalg.solve(X.T @ X, X.T)
    return Pe
