#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Submodule implementing generic solvers that can be used
to solve (non-linear) ODE systems of the form :

.. math::

    \frac{du}{dt} = f(u,t), \quad u(0) = u_0.

All those solvers are based on a :class:`DiffOp` base class,
implementing :

- the :math:`f(u,t)` evaluations,
- a solver for :math:`u-\alpha f(u,t)=rhs`, considering given :math:`\alpha,t,rhs`.

While the :math:`f(u,t)` evaluations must be implemented,
a default implementation of the solver for :math:`u-\alpha f(u,t)=rhs`
is provided in the base :class:`DiffOp` class.

    🛠️ Various specialized :class:`DiffOp` classes are implemented
    in the :class:`diffops` submodule.

The solvers implemented here discretizes
a time-step :math:`[t_0, t_0+\Delta{t}]` into **time nodes**
:math:`[t_0+\Delta{t}\tau_1, ..., t_0+\Delta{t}\tau_M]`
noted :math:`[t_1,\dots,t_M]`,
also called **stages** for RK methods, at which are defined the
**node solutions** :math:`u_m \simeq u(t_m)`.
And usually, the vector containing the node solutions
:math:`{\bf u} = [u_1,\dots,u_M]^T` satisfy a **all-at-once system** :

.. math::
    {\bf u} - \Delta{t}Q {\bf f} = {\bf u}_0,

where :math:`{\bf f} = [f(u_1, t_1),\dots,f(u_M,t_M)]^T` is the vector
with the evaluations of each node solutions
and :math:`{\bf u}_0` is a vector containing :math:`u_0` in each entry.
The :class:`CoeffSolver` allows to solve any ODE using this coefficient-based
approach, either directly if the :math:`Q` matrix is lower triangular,
or iteratively with SDC-based sweeps if :math:`Q` is a dense matrix.

----

An alternative solver approach relates all the node solutions using a
:math:`\phi` **representation** of a time-integrator,
*i.e* each node solution :math:`u_{m+1}` satisfies
the following relation :

.. math::

    u_{m+1} -\phi(u_0, u_1, ..., u_{m}, u_{m+1}) = u_0,

where :math:`\phi` is solely defined by the chosen time-integrator.
The system above can be solved node-by-node in a sequential approach,
or iteratively with a SDC-based approach.
It is implemented in the abstract :class:`PhiSolver` class,
that needs to be specialized by a child class implementing
the :math:`\phi` function.

    🛠️ Specialized :class:`PhiSolver` classes are implemented in the
    :class:`integrators` submodule.
"""
import numpy as np
import scipy.optimize as sco
from scipy.linalg import blas
import warnings

from qmat.solvers.dahlquist import Dahlquist
from qmat.lagrange import LagrangeApproximation


class DiffOp():
    r"""
    Base class for a differential operator :math:`f(u, t)` used in a generic ODE.

    It defines the evaluation of :math:`f(u, t)` at given :math:`u` and
    :math:`t` with a `evalF(u, t, out)` method, that put the result
    of the evaluation in the `out` array.

    Additionally, this class defines a default `fSolve` method that solves :

    .. math::

        u - \alpha f(u,t) = rhs

    for given :math:`\alpha`, :math:`t` and :math:`rhs`.
    This default method can be overridden by a more efficient specific
    method for a specific differential operator.

    Note
    ----
    Solutions are stored in N-dimensional :class:`numpy.ndarray`.

    Parameters
    ----------
    u0 : array-like
        The initial solution associated to the differential operator, to which
        is extracted the generic shape and datatype of :math:`u(t)` solutions.
    """
    def __init__(self, u0):
        for name in ["u0", "innerSolver"]:
            assert not hasattr(self, name), \
                f"{name} attribute is reserved for the base DiffOp class"
        self.u0 = np.asarray(u0)
        """Initial solution for the differential operator."""
        if self.u0.size < 1e3:
            self.innerSolver = sco.fsolve
            """Inner solver used in the default `fSolve` method."""
        else:
            self.innerSolver = sco.newton_krylov

    @property
    def uShape(self):
        """Shape of a :math:`u` solution, stored as numpy array."""
        return self.u0.shape

    @property
    def dtype(self):
        """Datatype of a :math:`u` solution, stored as numpy array."""
        return self.u0.dtype


    def evalF(self, u:np.ndarray, t:float, out:np.ndarray):
        """
        Evaluate :math:`f(u,t)` and store the result into `out`.

        Parameters
        ----------
        u : np.ndarray
            Input solution for the evaluation.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Output array in which is stored the evaluation.
        """
        raise NotImplementedError("evalF must be provided")


    def fSolve(self, a:float, rhs:np.ndarray, t:float, out:np.ndarray):
        r"""
        Solve :math:`u-\alpha f(u,t)=rhs` for given :math:`u,t,rhs`,
        using `out` as initial guess and storing the final result into it.

        Parameters
        ----------
        a : float
            The :math:`\alpha` coefficient.
        rhs : np.ndarray
            The right hand side.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Input-output array used as initial guess,
            in which is stored the solution.
        """
        def func(u:np.ndarray):
            """compute res = u - a*f(u,t) - rhs"""
            u = u.reshape(self.uShape)
            res = np.empty_like(u)
            self.evalF(u, t, out=res)
            res *= -a
            res += u
            res -= rhs
            return res.ravel()

        sol = self.innerSolver(func, out.ravel()).reshape(self.uShape)
        np.copyto(out, sol)


    @classmethod
    def test(cls, t0=0, dt=1e-1, eps=1e-3, instance=None):
        """
        Class method to test the `DiffOp` implementation.

        Parameters
        ----------
        t0 : float, optional
            Evaluation time to test the instance. The default is 0.
        dt : float, optional
            Time-step to test the `fSolve` method. The default is 1e-1.
        eps : float, optional
            Perturbation added in the expected solution to test the
            `fSolve` method. The default is 1e-3.
        instance :`DiffOp`, optional
            Instance to be tested. If not provided (`None`),
            an instance is created using the default constructor.
        """
        if instance is None:
            try:
                instance = cls()
            except:
                raise TypeError(f"{cls} cannot be instantiated with default parameters")

        u0 = instance.u0
        try:
            uEval = np.zeros_like(u0)
            instance.evalF(u=u0, t=t0, out=uEval)
        except:
            raise ValueError("evalF cannot be properly evaluated into an array like u0")

        try:
            uEval *= -dt
            uEval += u0
            uSolve = np.copy(u0)
            uSolve += eps*np.linalg.norm(uSolve, np.inf)
            instance.fSolve(a=dt, rhs=uEval, t=t0, out=uSolve)
        except:
            raise ValueError("fSolve cannot be properly evaluated into an array like u0")
        np.testing.assert_allclose(
            uSolve, u0, err_msg="fSolve does not satisfy the fixed-point problem with u0",
            atol=1e-15)

        # check for nan acceptation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uSolve[:] = np.nan
            instance.fSolve(a=dt, rhs=uEval, t=t0, out=uSolve)


class CoeffSolver():
    r"""
    Solve generic (non-linear) ODE system using :math:`Q`-coefficients with lower triangular form.

    It can be used to solve generic ODE systems of the form :

    .. math::

        \frac{du}{dt} = f(u,t), \quad u(0)=u_0.

    Parameters
    ----------
    diffOp : DiffOp
        Differential operator for the ODE.
    tEnd : float, optional
        Final simulation time. The default is 1.
    nSteps : int, optional
        Number of simulation time-steps. The default is 1.
    t0 : float, optional
        Initial simulation time. The default is 0.
    """
    def __init__(self, diffOp:DiffOp, tEnd=1, nSteps=1, t0=0):
        assert isinstance(diffOp, DiffOp)
        self.diffOp = diffOp
        """Differential Operator implementing :math:`f(u,t)`."""
        self.axpy = blas.get_blas_funcs('axpy', dtype=self.dtype)
        r"""BLAS-I function executing :math:`y=\alpha x + y` for any solution vectors :math:`x,y`."""

        self.t0 = t0
        """Initial simulation time."""
        self.tEnd = tEnd
        """Final simulation time."""
        self.nSteps = nSteps
        """Number of simulation time-steps"""
        self.dt = (tEnd-t0)/nSteps
        """Time-step size for the simulation"""

    @property
    def u0(self):
        """Initial solution for the problem"""
        return self.diffOp.u0

    @property
    def uShape(self):
        """Shape of the solution at a given time."""
        return self.diffOp.uShape

    @property
    def dtype(self):
        """Datatype of the solution at a given time."""
        return self.diffOp.dtype

    @property
    def times(self):
        """Time values for each time-step"""
        return np.linspace(self.t0, self.tEnd, self.nSteps+1)

    def evalF(self, u:np.ndarray, t:float, out:np.ndarray):
        """
        Wrapper for the `DiffOp` function evaluating :math:`f(u,t)`.

        Parameters
        ----------
        u : np.ndarray
            Input solution for the evaluation.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Output array in which is stored the evaluation.
        """
        self.diffOp.evalF(u, t, out)


    def fSolve(self, a:float, rhs:np.ndarray, t:float, out:np.ndarray):
        r"""
        Wrapper for the `DiffOp` function solving :math:`u-\alpha f(u,t) = rhs`.

        Parameters
        ----------
        a : float
            The :math:`\alpha` coefficient.
        rhs : np.ndarray
            The right hand side.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Input-output array used as initial guess,
            in which is stored the solution.
        """
        self.diffOp.fSolve(a, rhs, t, out)


    @staticmethod
    def lowerTri(Q:np.ndarray, strict=False):
        """
        Check if a 2D matrix is lower triangular.

        Parameters
        ----------
        Q : np.ndarray
            Matrix to check.
        strict : bool, optional
            Check for strictly lower triangular matrix. The default is False.

        Returns
        -------
        bool
            Is the matrix (strictly) lower triangular or not.
        """
        return np.allclose(np.triu(Q, k=0 if strict else 1), np.zeros(Q.shape))


    def solve(self, Q, weights, uNum=None, tInit=0):
        r"""
        Solve the ODE considering **lower-triangular** :math:`Q` coefficients.

        This is equivalent to the classical implementation of a generic
        Runge-Kutta method using its Butcher table.
        For each time-step, it defines a node solution (or stage)
        :math:`u_{m}` that is solved using previously computed
        node solution :

        .. math::

            u_{m} - \Delta{t}q_{m,m}f(u_m,t_m)
            = u_0 + \Delta{t}\sum_{j=1}^{m-1}q_{m,j}f(u_j, t_j),

        where :math:`t_m = t_0 + \tau_m` and :math:`q_{i,j}`
        are the coefficients :math:`Q`.
        Finally, the **step update** is done using all computed node
        solutions :

        .. math::
            u(t_0+\Delta{t}) \simeq
            u_0 + \sum_{m=1}^{M} \omega_{m} f(u_m, t_m),

        where :math:`\omega_{m}` are the weights associated to the
        :math:`Q`-coefficients.
        If no weights are provided, then it simply uses the last
        node solution for the step update :

        .. math::
            u(t_0+\Delta{t}) \simeq u_M

        Parameters
        ----------
        Q : np.2darray-like
            The **lower-triangular** :math:`Q`-coefficients matrix.
        weights : np.1darray-like
            The associated :math:\omega_{m}` weights. If not provided,
            use the last node solution for the update
            (requires :math:`\tau_{M} = 1`).
        uNum : np.ndarray, optional
            Array of shape `(nSteps+1,*uShape)`, that can be use
            to store the result and avoid creating it internally.
            The default is None.
        tInit : float, optional
            Initial time offset to be added to solver's own `t0` for
            successive `solve` calls. The default is 0.

        Returns
        -------
        uNum : np.ndarray
            Array of shape `(nSteps+1,*uShape)` that stores the solution at
            each time-step.
        """
        nNodes, Q, weights = Dahlquist.checkCoeff(Q, weights)

        assert self.lowerTri(Q), "lower triangular matrix Q expected for non-linear solver"
        Q = self.dt*Q
        if weights is not None:
            weights = self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0
        assert np.shape(uNum) == (self.nSteps+1, *self.uShape), \
            "user-provided uNum do not have the correct shape"

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        fEvals = np.zeros((nNodes, *self.uShape), dtype=self.dtype)

        times = self.times + tInit
        tau = Q.sum(axis=1)

        # time-stepping loop
        for i in range(self.nSteps):
            uNode = uNum[i+1]
            np.copyto(uNode, uNum[i])

            # loop on nodes (stages)
            for m in range(nNodes):
                tNode = times[i]+tau[m]

                # build RHS
                np.copyto(rhs, uNum[i])
                for j in range(m):
                    self.axpy(a=Q[m, j], x=fEvals[j], y=rhs)

                # solve node (if non-zero diagonal coefficient)
                if Q[m, m] != 0:
                    self.fSolve(a=Q[m, m], rhs=rhs, t=tNode, out=uNode)
                else:
                    np.copyto(uNode, rhs)

                # evalF on current store stage
                self.evalF(u=uNode, t=tNode, out=fEvals[m])

            # step update (if not, uNum[i+1] is already the last stage)
            if weights is not None:
                np.copyto(uNum[i+1], uNum[i])
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fEvals[m], y=uNum[i+1])

        return uNum


    def solveSDC(self, nSweeps, Q, weights, QDelta, uNum=None, tInit=0):
        r"""
        Solve the ODE with dense :math:`Q` coefficients using SDC sweeps.

        Considering a **lower-triangular** approximation :math:`Q_\Delta`
        of :math:`Q`, it performes for each time-step :math:`K` SDC sweeps :

        .. math::

            \begin{align}
            u_{m}^{k+1} - \Delta{t}q^\Delta_{m,m}f(u_m^{k+1},t_m)
                =&~ u_0 + \Delta{t}\sum_{j=1}^{M}q_{m,j}f(u_j^k, t_j) \\
            &+ \Delta{t}\sum_{j=1}^{m-1}q^\Delta_{m,j}f(u_j^{k+1},t_j)
            - \Delta{t}\sum_{j=1}^{m}q^\Delta_{m,j}f(u_j^{k},t_j),
            \end{align}

        where :math:`q^\Delta_{i,j}` and :math:`q_{i,j}` are the coefficients
        of :math:`Q_\Delta` and :math:`Q`, respectively.
        It uses a **copy initialization**, that is :math:`u_{m}^0 = u_0`.

        Finally, the **step update** is done using all computed node
        solutions :

        .. math::
            u(t_0+\Delta{t}) \simeq
            u_0 + \sum_{m=1}^{M} \omega_{m} f(u_m, t_m),

        where :math:`\omega_{m}` are the weights associated to the
        :math:`Q`-coefficients.
        If no weights are provided, then it simply uses the last
        node solution for the step update :

        .. math::
            u(t_0+\Delta{t}) \simeq u_M

        Parameters
        ----------
        nSweeps : int
            Number of SDC sweeps :math:`K`.
        Q : 2D array-like
            The dense :math:`Q` matrix.
        weights : 1D array-like
            The associated weights :math:`\omega_{m}` for the step update.
        QDelta : 2D array-like
            The lower-triangular :math:`Q_\Delta` matrix.
        uNum : np.ndarray, optional
            Array of shape `(nSteps+1,*uShape)`, that can be use
            to store the result and avoid creating it internally.
            The default is None.
        tInit : float, optional
            Initial time offset to be added to solver's own `t0` for
            successive `solve` calls. The default is 0.

        Returns
        -------
        uNum : np.ndarray
            Array of shape `(nSteps+1,*uShape)` that stores the solution at
            each time-step.
        """
        nNodes, Q, weights, QDelta, nSweeps = Dahlquist.checkCoeffSDC(Q, weights, QDelta, nSweeps)
        for qDelta in QDelta:
            assert self.lowerTri(qDelta), \
                "lower triangular matrices QDelta expected for non-linear SDC solver"

        Q, QDelta = self.dt*Q, self.dt*QDelta
        if weights is not None:
            weights = self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        fEvals = [np.zeros((nNodes, *self.uShape), dtype=self.dtype)
                  for _ in range(2)]

        times = self.times + tInit
        tau = Q.sum(axis=1)

        # time-stepping loop
        for i in range(self.nSteps):

            # copy initialization
            self.evalF(u=uNum[i], t=times[i], out=fEvals[0][0])
            np.copyto(fEvals[0][1:], fEvals[0][0])

            uNode = uNum[i+1]

            # loop on sweeps (iterations)
            for k in range(nSweeps):
                np.copyto(uNode, uNum[i])

                fK0 = fEvals[0]
                fK1 = fEvals[1]
                qDelta = QDelta[k]

                # loop on nodes (stages)
                for m in range(nNodes):
                    tNode = times[i] + tau[m]

                    # initialize RHS
                    np.copyto(rhs, uNum[i])

                    # add quadrature terms
                    for j in range(nNodes):
                        self.axpy(a=Q[m, j], x=fK0[j], y=rhs)

                    # add correction terms (from previous nodes)
                    for j in range(m):
                        self.axpy(a= qDelta[m, j], x=fK1[j], y=rhs)
                        self.axpy(a=-qDelta[m, j], x=fK0[j], y=rhs)

                    # diagonal term (current node)
                    if qDelta[m, m] != 0:
                        self.axpy(a=-qDelta[m, m], x=fK0[m], y=rhs)
                        self.fSolve(a=qDelta[m, m], rhs=rhs, t=tNode, out=uNode)
                    else:
                        np.copyto(uNode, rhs)

                    # evalF on current node
                    self.evalF(u=uNode, t=tNode, out=fK1[m])

                # invert fK0 and fK1 for the next sweep
                fEvals[0], fEvals[1] = fEvals[1], fEvals[0]

            # step update (if not, uNum[i+1] is already the last stage)
            if weights is not None:
                np.copyto(uNum[i+1], uNum[i])
                for m in range(nNodes):
                    self.axpy(a=weights[m], x=fK1[m], y=uNum[i+1])

        return uNum


class PhiSolver(CoeffSolver):
    r"""
    Solve generic (non-linear) ODE system using :math:`\phi` representation of time-integration solvers.

    It consider the following ODE :

    .. math::
        \frac{du}{dt} = f(u,t),

    and compute for each step the solution on **time nodes** :math:`\tau_1, ..., \tau_M`
    by soving the following system :

    .. math::

        u_{m+1} -\phi(u_0, u_1, ..., u_{m}, u_{m+1}) = u_0.

    It uses then per default the last node solution :math:`u_{M}` as initial
    solution for the next step.

    ⚙️ Requires the implementation of an `evalPhi` method that evaluates
    the :math:`\phi` function.
    Also, a default `phiSolve` method is implemented, that solves
    the system above, and can be overridden for specific time-integrator
    (in particular for explicit time-integrators).
    Finally, it implements a default `stepUpdate` method that setup the
    next time-step using the last time-node solution.

    Parameters
    ----------
    diffOp : DiffOp
        Differential operator for the ODE.
    nodes : 1D array-like
        The time nodes :math:`\tau_1, ..., \tau_M`.
    tEnd : float, optional
        Final simulation time. The default is 1.
    nSteps : int, optional
        Number of simulation time-steps. The default is 1.
    t0 : float, optional
        Initial simulation time. The default is 0.
    """
    def __init__(self, diffOp:DiffOp, nodes, tEnd=1, nSteps=1, t0=0):
        super().__init__(diffOp, tEnd, nSteps, t0)
        self.nodes = np.asarray(nodes, dtype=float)
        """Time nodes for each time-step of the time-integrator."""

    @property
    def nNodes(self):
        """Number of time-nodes"""
        return self.nodes.size


    def evalPhi(self, uVals, fEvals, out, t0=0):
        r"""
        Evaluate the :math:`\phi` operator on time-node up to :math:`u_{m+1}`.

        Considering :math:`u_0, u_1, \dots, u_{m+1}`,
        if evaluates :

        .. math::

            \phi(u_0, u_1, ..., u_{m}, u_{m+1}),

        and store its value into the output vector `out`.
        It also takes the node evaluation
        :math:`f(u_0,t_0),f(u_1,\tau_1),...,f(u_{m},\tau_{m})`
        as arguments, in order to avoid any additional :math:`f(u,t)`
        evaluations.

        Parameters
        ----------
        uVals : list[np.ndarray] of size :math:`m+2`
            The :math:`m+1` time-node solutions + the initial solution :math:`u_0`.
        fEvals : list[np.ndarray] of size :math:`m+1` or :math:`m+1`
            The :math:`f(u,t)` evaluations at each time nodes (+ initial solution),
            up to time-node :math:`m`.
            It can eventually contain a pre-computed :math:`f_{m+1}`
            to spare one :math:`f(u,t)` evaluation.
        out : np.ndarray
            Array used to store the evaluation.
        t0 : float, optional
            Initial step time. The default is 0.
        """
        raise NotImplementedError(
            "specialized PhiSolver must implement its evalPhi method")


    def phiSolve(self, uPrev, fEvals, out, rhs=0, t0=0):
        r"""
        Solve the node update at given time-node :math:`\tau_{m+1}`.

        Considering :math:`m+1` previous known node solutions
        :math:`u_0, u_1, ..., u_{m}`, it solves the following system :

        .. math::

            u -\phi(u_0, u_1, ..., u_{m}, u)
            = rhs,

        where the value given in `out` is used as **initial guess** and
        to **store the computed solution**.
        It also takes as argument the :math:`f` evaluations
        :math:`f_0, f_1, ..., f_{m}` to avoid supplementar re-computing those.

        Parameters
        ----------
        uPrev : list[np.ndarray] of size :math:`m+1`
            The previous node solutions :math:`u_0, u_1, ..., u_{m}`.
        fEvals : list[np.ndarray] of size :math:`m+1`
            Evaluations of previous node solutions :math:`f_0, f_1, ..., f_{m}`.
        out : np.ndarray
            Array with the initial guess, used to store the final solution.
        rhs : np.ndarray or float, optional
            Right hand side used to solve the equation above.
            The default is 0.
        t0 : float, optional
            Initial step size. The default is 0.
        """
        assert len(fEvals) == len(uPrev)

        def func(u:np.ndarray):
            u = u.reshape(self.uShape)
            res = np.empty_like(u)
            self.evalPhi([*uPrev, u], fEvals, out=res, t0=t0)
            res *= -1
            res += u
            res -= rhs
            return res.ravel()

        sol = self.diffOp.innerSolver(func, out.ravel()).reshape(self.uShape)
        np.copyto(out, sol)


    def stepUpdate(self, u0, uNodes, fEvals, out):
        r"""
        Update end-step solution to be used as initial guess for next step.

        Note
        ----
        This method has to ensures that fEvals[0] contains the :math:`f(u,t)`
        evaluation of the next step initial solution.

        Parameters
        ----------
        u0 : np.ndarray
            Initial solution for the current step.
        uNodes : list[np.ndarray]
            Precomputed node solutions :math:`u_1,\dots,u_M`.
        fEvals : list[np.ndarray]
            Precomputed node evaluation :math:`f_1,\dots,f_M`.
        out : np.ndarray
            Output array to store the result.
        """
        assert self.nodes[-1] == 1
        np.copyto(out, uNodes[-1])
        fEvals[0], fEvals[-1] = fEvals[-1], fEvals[0]


    def solve(self, uNum=None, tInit=0):
        r"""
        Solve using sequential computation of node solutions for each step,
        using the relation :

        .. math::

            u_{m+1} -\phi(u_0, u_1, ..., u_{m}, u_{m+1}, f_0, f_1, ..., f_{m})
            = u_0.

        and the step update to compute :math:`u(t_0+\Delta_t)` using all
        computed node solutions.


        Parameters
        ----------
        uNum : np.ndarray, optional
            Array of shape `(nSteps+1,*uShape)`, that can be use
            to store the result and avoid creating it internally.
            The default is None.
        tInit : float, optional
            Initial time offset to be added to solver's own `t0` for
            successive `solve` calls. The default is 0.

        Returns
        -------
        uNum : np.ndarray
            Array of shape `(nSteps+1,*uShape)` that stores the solution at
            each time-step.
        """
        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        uNodes = np.zeros((self.nNodes, *self.uShape), dtype=self.dtype)
        fEvals = [np.zeros(self.uShape, dtype=self.dtype)
                  for _ in range(self.nNodes+1)]
        self.evalF(uNum[0], self.t0, out=fEvals[0])

        times = self.times + tInit
        tau = self.dt*self.nodes

        # time-stepping loop
        for i in range(self.nSteps):

            # initialize first node with starting value for step
            np.copyto(uNodes[0], uNum[i])

            # loop on nodes
            for m in range(self.nNodes):
                self.phiSolve(
                    [uNum[i], *uNodes[:m]], fEvals[:m+1], rhs=uNum[i], out=uNodes[m], t0=times[i])
                self.evalF(u=uNodes[m], t=times[i]+tau[m], out=fEvals[m+1])

            # step update
            self.stepUpdate(uNum[i], uNodes, fEvals, out=uNum[i+1])

        return uNum


    def solveSDC(self, nSweeps, Q=None, weights=None, uNum=None, tInit=0):
        r"""
        Solve the ODE with dense :math:`Q` coefficients using SDC sweeps.

        Considering a **lower-triangular** approximation :math:`Q_\Delta`
        of :math:`Q`, it performes for each time-step :math:`K` SDC sweeps :

        .. math::

            u_{m}^{k+1} - \phi_m^{k+1}
                = u_0 + \Delta{t}\sum_{j=1}^{M}q_{m,j}f(u_j^k, t_j)
                - \phi_m^k,

        where
        :math:`\phi_m^k:=\phi(u_0,u_1^k,\dots,u_m^k,f_0,f_1^k,\dots,f_{m-1}^k)`
        and :math:`q_{i,j}` are the coefficients of the :math:`Q` matrix.
        It uses a **copy initialization**, that is :math:`u_{m}^0 = u_0`.

            💡 If we consider that :math:`\phi_m^{k}` is like
            a coarse solver applied on iteration :math:`k` and
            :math:`u_0 + \Delta{t}\sum_{j=1}^{M}q_{m,j}f(u_j^k, t_j)` is like
            a fine solver applied to iteration :math:`k`,
            then the SDC correction above furiously resemble to
            a **Parareal iteration** 👻 👻 👻

        Finally, the **step update** is done using all computed node
        solutions :

        .. math::
            u(t_0+\Delta{t}) \simeq
            u_0 + \sum_{m=1}^{M} \omega_{m} f(u_m, t_m),

        where :math:`\omega_{m}` are the weights associated to the
        :math:`Q`-coefficients.
        If weights are not used (`weights=False`),
        then it simply uses the last node solution for the step update :

        .. math::
            u(t_0+\Delta{t}) \simeq u_M

        Parameters
        ----------
        nSweeps : int
            Number of SDC sweeps :math:`K`.
        Q : 2D array-like, optional
            The dense :math:`Q` matrix.
            If not provided, automatically computed using the
            :class:`LagrangeApproximation` class and the solver nodes.
        weights : 1D array-like, optional
            The associated weights :math:`\omega_{m}` for the step update.
            If not provided, automatically computed using the
            :class:`LagrangeApproximation` class and the solver nodes.
        uNum : np.ndarray, optional
            Array of shape `(nSteps+1,*uShape)`, that can be use
            to store the result and avoid creating it internally.
            The default is None.
        tInit : float, optional
            Initial time offset to be added to solver's own `t0` for
            successive `solve` calls. The default is 0.

        Returns
        -------
        uNum : np.ndarray
            Array of shape `(nSteps+1,*uShape)` that stores the solution at
            each time-step.
        """
        if Q is None or weights is True:
            approx = LagrangeApproximation(self.nodes)
        if Q is None:
            Q = approx.getIntegrationMatrix([(0, tau) for tau in self.nodes])
        if weights is True:
            weights = approx.getIntegrationMatrix([(0, 1)]).ravel()
        if weights is False:
            weights = None
        nNodes, Q, weights = Dahlquist.checkCoeff(Q, weights)
        assert nNodes == self.nNodes, "solver and Q do not have the same number of nodes"
        assert np.allclose(Q.sum(axis=1), self.nodes), "solver and Q do not have the same nodes"
        Q = self.dt*Q
        if weights is not None:
            weights = self.dt*weights

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        uNodes = [np.zeros((self.nNodes, *self.uShape), dtype=self.dtype)
                  for _ in range(2)]
        fEvals = [[np.zeros(self.uShape, dtype=self.dtype)
                   for _ in range(self.nNodes+1)]
                  for _ in range(2)]

        times = self.times + tInit
        tau = self.dt*self.nodes

        # time-stepping loop
        for i in range(self.nSteps):

            # copy initialization
            np.copyto(uNodes[0], uNum[i])
            self.evalF(uNum[i], times[i], out=fEvals[0][0])
            np.copyto(fEvals[1][0], fEvals[0][0])   # u_0^{1} = u_0^{0}
            for m in range(self.nNodes):
                np.copyto(fEvals[0][m+1], fEvals[0][0])  # u_m^{k} = u_0^{0}

            uTmp = uNum[i+1]    # use next step as buffer for k correction term

            # loop on sweeps (iterations)
            for _ in range(nSweeps):

                uK0, uK1 = uNodes
                fK0, fK1 = fEvals

                # loop on nodes (stages)
                for m in range(self.nNodes):

                    # initialize RHS
                    np.copyto(rhs, uNum[i])

                    # add quadrature terms
                    fK = fK0[1:]  # note : ignore f(u0) term in fK0
                    for j in range(self.nNodes):
                        self.axpy(a=Q[m, j], x=fK[j], y=rhs)

                    # substract k correction term
                    self.evalPhi(
                        [uNum[i], *uK0[:m+1]], fK0[:m+2], out=uTmp, t0=times[i])
                    rhs -= uTmp

                    # solve with k+1 correction
                    self.phiSolve(
                        [uNum[i], *uK1[:m]], fK1[:m+1], out=uK1[m], rhs=rhs, t0=times[i])

                    # evalF on k+1 node solution
                    self.evalF(uK1[m], t=times[i]+tau[m], out=fK1[m+1])

                # invert uK0/fK0 and uK1/fK1 for next sweep
                fEvals[0], fEvals[1] = fEvals[1], fEvals[0]
                uNodes[0], uNodes[1] = uNodes[1], uNodes[0]

            # step update
            if weights is not None:
                np.copyto(uNum[i+1], uNum[i])
                fK = fK1[1:]  # note : ignore f(u0) term in fK0
                for m in range(self.nNodes):
                    self.axpy(a=weights[m], x=fK[m], y=uNum[i+1])
            else:
                self.stepUpdate(uNum[i], uNodes[0], fEvals[0], out=uNum[i+1])

        return uNum
