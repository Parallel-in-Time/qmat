#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for Barycentric Lagrange Approximation, based on `[Berrut & Trefethen, 2004] <https://doi.org/10.1137/S0036144502417715>`_.
Allows to easily build integration / interpolation / derivative matrices, from any list of node points.

Examples
--------
>>> # Base usage to generate a quadrature matrix
>>> from qmat.lagrange import LagrangeApproximation, np
>>>
>>> grid = np.linspace(0, 1, num=5)
>>> approx = LagrangeApproximation(grid)
>>> Q = approx.getIntegrationMatrix([(0, tau) for tau in grid])
>>>
>>> # Interpolation
>>> fGrid = np.linspace(0, 1, num=200)
>>> u = np.exp(grid)
>>> P = approx.getInterpolationMatrix(fGrid)
>>> uFine = P @ u
>>>
>>> # Alternative interpolation using the object as a function
>>> uFine = approx(fGrid)
>>>
>>> # Derivative
>>> D = approx.getDerivativeMatrix()
>>> du = D @ u
"""
import numpy as np
from scipy.special import roots_legendre


def computeFejerRule(n):
    """
    Compute a Fejer rule of the first kind, using DFT `[Waldvogel, 2006] <https://link.springer.com/article/10.1007/s10543-006-0045-4>`_.
    Inspired from quadpy (https://github.com/nschloe/quadpy @Nico_Schlömer)

    Parameters
    ----------
    n : int
        Number of points for the quadrature rule.

    Returns
    -------
    nodes : np.1darray(n)
        The nodes of the quadrature rule
    weights : np.1darray(n)
        The weights of the quadrature rule.
    """
    # Initialize output variables
    n = int(n)
    nodes = np.empty(n, dtype=float)
    weights = np.empty(n, dtype=float)

    # Compute nodes
    theta = np.arange(1, n + 1, dtype=float)[-1::-1]
    theta *= 2
    theta -= 1
    theta *= np.pi / (2 * n)
    np.cos(theta, out=nodes)

    # Compute weights
    # -- Initial variables
    N = np.arange(1, n, 2)
    lN = len(N)
    m = n - lN
    K = np.arange(m)
    # -- Build v0
    v0 = np.concatenate([
        2 * np.exp(1j * np.pi * K / n) / (1 - 4 * K**2),
        np.zeros(lN + 1)])
    # -- Build v1 from v0
    v1 = np.empty(len(v0) - 1, dtype=complex)
    np.conjugate(v0[:0:-1], out=v1)
    v1 += v0[:-1]
    # -- Compute inverse fourier transform
    w = np.fft.ifft(v1)
    if max(w.imag) > 1.0e-15:
        raise ValueError(
            f'Max imaginary value to important for ifft: {max(w.imag)}')
    # -- Store weights
    weights[:] = w.real

    return nodes, weights


class LagrangeApproximation(object):
    r"""
    Class approximating any function on a given set of points using barycentric
    Lagrange interpolation.

    Let note :math:`(t_j)_{0\leq j<n}` the set of points, then any scalar
    function :math:`f` can be approximated by the barycentric formula :

    .. math::
        p(x) =
        \frac{\displaystyle \sum_{j=0}^{n-1}\frac{w_j}{x-x_j}f_j}
        {\displaystyle \sum_{j=0}^{n-1}\frac{w_j}{x-x_j}},

    where :math:`f_j=f(t_j)` and

    .. math::
        w_j = \frac{1}{\prod_{k\neq j}(x_j-x_k)}

    are the barycentric weights.
    The theory and implementation is inspired from [1]_.

    Parameters
    ----------
    points : list, tuple or np.1darray
        The given interpolation points, no specific scaling, but must be
        ordered in increasing order.
    weightComputation : str, optional
        Algorithm used to compute the barycentric weights. Can be :

        - 'FAST' : uses the analytic formula (unstable for large number of points)
        - 'STABLE' : uses logarithmic difference and scaling of the weights
        - 'CHEBFUN' : uses the same approach as in the chebfun package

        The default is 'AUTO' : it tries the 'FAST' algorithm, and if an
        overflow is detected, it switches to the 'STABLE' algorithm.
    scaleRef : str, optional
        Scaling used in the 'STABLE' algorithm for weight computation.
        Can be :

        - 'ZERO' : scaling based on the weight for the value closest to :math:`t=0`.
        - 'MAX' : scaling based on the maximum weight value.

        The default is 'MAX'.
    duplicates : str
            Which strategy to use in case of duplicated values within the interpolation
            points. Can be :

            - 'USE_LEFT' : uses the first value from the left in the values vector
            - 'USE_RIGHT' : uses the first value from the right in the values vector

            The default is 'USE_LEFT'.
    fValues : list, tuple or np.1darray
        Function values to be used when evaluating the LagrangeApproximation as a function

    Attributes
    ----------
    points : np.1darray
        The interpolating points
    weights : np.1darray
        The associated barycentric weights
    nPoints : int (property)
        The number of points, can also be retrieve with `n` (legacy alias)
    uniquePoints : np.1darray
        The unique interpolating points.
        When there is no duplicates, points == uniquePoints.
    nUniquePoints : int (property)
        The number of unique points
    duplicates : str
        The strategy used when there is duplicated interpolation points

    References
    ----------
    .. [1] Berrut, J. P., & Trefethen, L. N. (2004).
        "Barycentric Lagrange interpolation." SIAM review, 46(3), 501-517.
        URL: https://doi.org/10.1137/S0036144502417715
    """

    def __init__(self, points,
                 weightComputation='AUTO', scaleWeights=False, scaleRef='MAX',
                 duplicates="USE_LEFT", fValues=None):
        points = np.asarray(points, dtype=float).ravel()

        uniques = np.unique(points)
        if points.size != uniques.size:
            self.points, self.uniquePoints = points, uniques
            self.duplicates = duplicates
            self._setupDuplicates()
        else:
            self.points = self.uniquePoints = points
            self._handleDuplicates = self._passThrough

        points = uniques  # require unique points for weight computation

        diffs = points[:, None] - points[None, :]
        diffs[np.diag_indices_from(diffs)] = 1

        def analytic(diffs):
            # Fast implementation (unstable for large number of points)
            invProd = np.prod(diffs, axis=1)
            invProd **= -1
            return invProd

        def logScale(diffs):
            # Implementation using logarithmic difference and scaling
            sign = np.sign(diffs).prod(axis=1)
            wLog = -np.log(np.abs(diffs)).sum(axis=1)
            if scaleRef == 'ZERO':
                wScale = wLog[np.argmin(np.abs(points))]
            elif scaleRef == 'MAX':
                wScale = wLog.max()
            else:
                raise NotImplementedError(f'scaleRef={scaleRef}')
            invProd = np.exp(wLog - wScale)
            invProd *= sign
            return invProd

        def chebfun(diffs):
            # Implementation used in chebfun
            diffs *= 4 / (points.max() - points.min())
            sign = np.sign(diffs).prod(axis=1)
            vv = np.exp(np.log(np.abs(diffs)).sum(axis=1))
            invProd = (sign * vv)
            invProd **= -1
            invProd /= np.linalg.norm(invProd, np.inf)
            return invProd

        if weightComputation == 'AUTO':
            with np.errstate(divide='raise', over='ignore'):
                try:
                    invProd = analytic(diffs)
                except FloatingPointError:
                    invProd = logScale(diffs)
        elif weightComputation == 'FAST':
            invProd = analytic(diffs)
        elif weightComputation == 'STABLE':
            invProd = logScale(diffs)
        elif weightComputation == 'CHEBFUN':
            invProd = chebfun(diffs)
        else:
            raise NotImplementedError(
                f'weightComputation={weightComputation}')
        weights = invProd
        if scaleWeights:
            weights /= np.max(np.abs(weights))

        # Store weights
        self.weights = weights
        self.weightComputation = weightComputation

        # Store function values if provided
        if fValues is not None:
            fValues = np.asarray(fValues)
            if fValues.shape != self.points.shape:
                raise ValueError(f'fValues {fValues.shape} has not the correct shape: {self.points.shape}')
        self.fValues = fValues


    def __call__(self, t, fValues=None):
        if fValues is None: fValues=self.fValues
        assert fValues is not None, "cannot evaluate polynomial without fValues"
        t = np.asarray(t)
        fValues = np.asarray(fValues)
        values = self.getInterpolationMatrix(t.ravel()).dot(fValues)
        values.shape = t.shape
        return values


    def _setupDuplicates(self):
        """Check the duplicates parameters"""
        # TODO : allow some convex combinations between duplicates
        if self.duplicates not in ["USE_LEFT", "USE_RIGHT"]:
            raise NotImplementedError(f"duplicates={self.duplicates}")

        values, indices, self._invIdx = np.unique(
            self.points, return_index=True, return_inverse=True)

        if self.duplicates == "USE_LEFT":
            self._nnzIdx = indices

        if self.duplicates == "USE_RIGHT":
            self._nnzIdx = [
                np.max(np.where(self.points == pts)) for pts in values]

        self._zerIdx = np.setdiff1d(np.arange(self.nPoints), self._nnzIdx)

    def _handleDuplicates(self, matrix):
        """Modify a matrix when there is duplicates"""
        out = matrix[:, self._invIdx]
        out[:, self._zerIdx] = 0
        return out

    def _passThrough(self, matrix):
        """Simply pass through a matrix when no duplicates"""
        return matrix


    @property
    def nPoints(self)->int:
        """The number of points"""
        return self.points.size

    n = nPoints
    """Legacy alias for nPoints"""

    @property
    def nUniquePoints(self)->int:
        """The number of unique points"""
        return self.uniquePoints.size

    @property
    def hasDuplicates(self)->bool:
        """Wether the points have duplicates or not"""
        return self.nPoints > self.nUniquePoints

    def getInterpolationMatrix(self, times, duplicates=True):
        r"""
        Compute the interpolation matrix for a given set of discrete "time"
        points.

        For instance, if we note :math:`\vec{f}` the vector containing the
        :math:`f_j=f(t_j)` values, and :math:`(\tau_m)_{0\leq m<M}`
        the "time" points where to interpolate.
        Then :math:`I[\vec{f}]`, the vector containing the interpolated
        :math:`f(\tau_m)` values, can be obtained using :

        .. math::
            I[\vec{f}] = P_{Inter} \vec{f},

        where :math:`P_{Inter}` is the interpolation matrix returned by this
        method.

        Parameters
        ----------
        times : list-like or np.1darray
            The discrete "time" points where to interpolate the function.
        duplicates : bool
            Wether or not take into account duplicates in the points.
            This has no impact if all interpolating points are distinct.
            Default is True.

        Returns
        -------
        P : np.2darray(M, n)
            The interpolation matrix, with :math:`M` rows (size of the **times**
            parameter) and :math:`n` columns.

        """
        # Compute difference between times and Lagrange points
        times = np.asarray(times)
        assert times.ndim == 1, "times is not a 1D array"
        with np.errstate(divide='ignore'):
            iDiff = 1 / (times[:, None] - self.uniquePoints[None, :])

        # Find evaluated positions that coincide with one Lagrange point
        concom = (iDiff == np.inf) | (iDiff == -np.inf)
        i, j = np.where(concom)

        # Replace iDiff by one on those lines to get a simple copy of the value
        iDiff[i, :] = concom[i, :]

        # Compute interpolation matrix using weights
        P = iDiff * self.weights
        P /= P.sum(axis=-1)[:, None]

        if duplicates:
            P = self._handleDuplicates(P)

        return P


    def getIntegrationMatrix(self, intervals, numQuad='FEJER', duplicates=True):
        r"""
        Compute the integration matrix for a given set of intervals.

        For instance, if we note :math:`\vec{f}` the vector containing the
        :math:`f_j=f(t_j)` values, and
        :math:`(\tau_{m,left}, \tau_{m,right})_{0\leq m<M}` the different
        interval where the function should be integrated using the barycentric
        interpolant polynomial.
        Then :math:`\Delta[\vec{f}]`, the vector containing the approximations
        of

        .. math::
            \int_{\tau_{m,left}}^{\tau_{m,right}} f(t)dt,

        can be obtained using :

        .. math::
            \Delta[\vec{f}] = P_{Integ} \vec{f},

        where :math:`P_{Integ}` is the interpolation matrix returned by this
        method.

        Parameters
        ----------
        intervals : list of pairs
            A list of all integration intervals.
        numQuad : str, optional
            Quadrature rule used to integrate the interpolant barycentric
            polynomial. Can be :

            - 'LEGENDRE_NUMPY' : Gauss-Legendre rule from Numpy
            - 'LEGENDRE_SCIPY' : Gauss-Legendre rule from Scipy
            - 'FEJER' : internally implemented Fejer-I rule

            The default is 'FEJER'.
        duplicates : bool
            Wether or not take into account duplicates in the points.
            This has no impact if all interpolating points are distinct.
            Default is True.

        Returns
        -------
        Q : np.2darray(M, n)
            The integration matrix, with :math:`M` rows (number of intervals)
            and :math:`n` columns.
        """
        intervals = np.array(intervals)
        assert intervals.ndim == 2, "intervals is not a 2D array"
        assert intervals.shape[1] == 2, "intervals must contains only pairs"

        if numQuad == 'LEGENDRE_NUMPY':
            # Legendre gauss rule, integrate exactly polynomials of deg. (2n-1)
            iNodes, iWeights = np.polynomial.legendre.leggauss((self.n + 1) // 2)
        elif numQuad == 'LEGENDRE_SCIPY':
            # Using Legendre scipy implementation
            iNodes, iWeights = roots_legendre((self.n + 1) // 2)
        elif numQuad == 'FEJER':
            # Fejer-I rule, integrate exactly polynomial of deg. n-1
            iNodes, iWeights = computeFejerRule(self.n - ((self.n + 1) % 2))
        else:
            raise NotImplementedError(f'numQuad={numQuad}')

        # Compute quadrature nodes for each interval
        aj, bj = intervals[:, 0][:, None], intervals[:, 1][:, None]
        tau, omega = iNodes[None, :], iWeights[None, :]
        tEval = (bj - aj) / 2 * tau + (bj + aj) / 2

        # Compute the integrand function on nodes
        integrand = self.getInterpolationMatrix(
            tEval.ravel(), duplicates=False).T.reshape(
            (-1,) + tEval.shape)

        # Apply quadrature rule to integrate
        integrand *= omega
        integrand *= (bj - aj) / 2
        Q = integrand.sum(axis=-1).T

        if duplicates:
            Q = self._handleDuplicates(Q)

        return Q

    def getDerivativeMatrix(self, order=1, duplicates=True):
        r"""
        Generate derivative matrix of first or second order (or both) based on
        the Lagrange interpolant.
        The first order differentiation matrix :math:`D^{(1)}` approximates

        .. math::
            D^{(1)} u \simeq \frac{du}{dx}

        on the interpolation points. The formula is :

        .. math::
            D^{(1)}_{ij} = \frac{w_j/w_i}{x_i-x_j}

        for :math:`i \neq j` and

        .. math::
            D^{(1)}_{jj} = -\sum_{i \neq j} D^{(1)}_{ij}`

        The second order differentiation matrix :math:`D^{(2)}` approximates

        .. math::
            D^{(2)} u \simeq \frac{d^2u}{dx^2}

        on the interpolation points. The formula is :

        .. math::
            D^{(1)}_{ij} = -2\frac{w_j/w_i}{x_i-x_j}\left[
                \frac{1}{x_i-x_j} + \sum_{k \neq i}\frac{w_k/w_i}{x_i-x_k}
                \right]

        for :math:`i \neq j` and

        .. math::
            D^{(2)}_{jj} = -\sum_{i \neq j} D^{(2)}_{ij}

        ⚠️ If you want a derivative matrix with many points (~1000 or more),
        favor the use of `weightComputation="STABLE"` when initializing
        the `LagrangeApproximation` object. If not, some (very small) weights
        could be approximated by zeros, which would make the computation
        of the derivative matrices fail ...

        Note
        ----
        There is a typo in the formula for :math:`D^{(2)}` given in the paper
        of Berrut and Trefethen. The formula above is the correct one.

        Parameters
        ----------
        order : int or str, optional
            The order of the derivative matrix, use "ALL" to retrieve both.
            The default is 1.
        duplicates : bool
            Wether or not take into account duplicates in the points.
            This has no impact if all interpolating points are distinct.
            Default is True.

        Returns
        -------
        D : np.2darray or tuple of np.2darray
            Derivative matrix. If order="ALL", return a tuple containing all
            derivative matrices in increasing derivative order.
        """
        if order not in [1, 2, "ALL"]:
            raise NotImplementedError(f"order={order}")
        w = self.weights
        x = self.uniquePoints

        with np.errstate(divide='ignore'):
            iDiff = 1 / (x[:, None] - x[None, :])
        iDiff[np.isinf(iDiff)] = 0

        base = w[None, :]/w[:, None]
        base *= iDiff

        if order in [1, "ALL"]:
            D1 = base.copy()
            np.fill_diagonal(D1, -D1.sum(axis=-1))
            if duplicates:
                D1 = self._handleDuplicates(D1)
        if order in [2, "ALL"]:
            D2 = -2*base
            D2 *= iDiff + base.sum(axis=-1)[:, None]
            np.fill_diagonal(D2, -D2.sum(axis=-1))
            if duplicates:
                D2 = self._handleDuplicates(D2)

        if order == 1:
            return D1
        elif order == 2:
            return D2
        else:
            return D1, D2

    def getDerivationMatrix(self, *args, **kwargs):
        import warnings
        warnings.warn("Function `getDerivationMatrix` is deprecated. Use `getDerivativeMatrix` instead!", DeprecationWarning)
        return self.getDerivativeMatrix(*args, **kwargs)


def getSparseInterpolationMatrix(inPoints, outPoints, order):
    """
    Get a sparse interpolation matrix from `inPoints` to `outPoints` of order
    `order` using barycentric Lagrange interpolation.

    The matrix will have `order` entries per line, and tends to be banded when
    both `inPoints` and `outPoints` are equispaced and cover the same interval.

    Parameters
    ----------
        inPoints : np.1darray
            The points you want to interpolate from
        outPoints : np.1darray
            The points you want to interpolate to
        order : int
            Order of the interpolation

    Returns
    -------
    A : scipy.sparse.csc_matrix(len(outPoints), len(inPoints))
        Sparse interpolation matrix
    """
    import scipy.sparse as sp

    assert order <= len(inPoints), f'Cannot interpolate {len(inPoints)} to order {order}! Please reduce order'

    A = sp.lil_matrix((len(outPoints), len(inPoints)))
    lastInterpolationLine = None
    lastClosestPoints = None

    for i in range(len(outPoints)):
        closestPointsIdx = np.sort(np.argsort(np.abs(inPoints - outPoints[i]))[:order])
        closestPoints = inPoints[closestPointsIdx] - outPoints[i]

        if lastClosestPoints is not None and np.allclose(closestPoints, lastClosestPoints):
            interpolationLine = lastInterpolationLine
        else:
            interpolator = LagrangeApproximation(points = closestPoints)
            interpolationLine = interpolator.getInterpolationMatrix([0])[0]

            lastInterpolationLine = interpolationLine
            lastClosestPoints = closestPoints

        A[i, closestPointsIdx] = interpolationLine

    return A.tocsc()
