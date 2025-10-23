from abc import ABC, abstractmethod
import numpy as np


class DESolver(ABC):

    @abstractmethod
    def evalF(self, u: np.ndarray, t: float) -> np.ndarray:
        """Evaluate the right-hand side of the equation.

        Parameters
        ----------
        u : np.ndarray
            Array of shape (N,) representing the solution at the current time step.
        t : float
            Current timestamp.
        """
        pass

    def evalF1(self, u: np.ndarray, t: float) -> np.ndarray:
        """Evaluate the 1st term of the equation

        Parameters
        ----------
        u : np.ndarray
            Array of shape (N,) representing the solution at the current time step.
        t : float
            Current timestamp.
        """
        raise Exception("TODO: `evalF1` not implemented for this DE solver.")

    def evalF2(self, u: np.ndarray, t: float) -> np.ndarray:
        """Evaluate the 2nd term of the equation

        Parameters
        ----------
        u : np.ndarray
            Array of shape (N,) representing the solution at the current time step.
        t : float
            Current timestamp.
        """
        raise Exception("TODO: `evalF2` not implemented for this DE solver.")

    # This is optional since not every DE might have a solver for backward Euler
    def fSolve(self, rhs: np.ndarray, dt: float, t: float) -> np.ndarray:
        """Solve the right-hand side of an equation implicitly.

        # Solving this equation implicitly...
        u_t = f(u, t)

        # ... means to u_new
        u_new - dt * F(u_new, t + dt) = rhs

        Parameters
        ----------
        rhs : np.ndarray
            Right hand as given above.
        t : float
            Current timestamp.
            Future one will be computed as `t + dt`
        dt : float
            Time step size.
        """
        raise Exception("TODO: Implicit solver not implemented for this DE solver.")

    def fSolve1(self, rhs: np.ndarray, dt: float, t: float) -> np.ndarray:
        """Solve the right-hand side of the 1st equation implicitly.

        # Solving this equation implicitly...
        u_t = f(u, t)

        # ... means to u_new
        u_new - dt * F(u_new, t + dt) = rhs

        Parameters
        ----------
        rhs : np.ndarray
            Right hand as given above.
        t : float
            Current timestamp.
            Future one will be computed as `t + dt`
        dt : float
            Time step size.
        """
        raise Exception("TODO: `fSolve1` not implemented for this DE solver.")

    def fSolve2(self, rhs: np.ndarray, dt: float, t: float) -> np.ndarray:
        """Solve the right-hand side of the 1st equation implicitly.

        # Solving this equation implicitly...
        u_t = f(u, t)

        # ... means to u_new
        u_new - dt * F(u_new, t + dt) = rhs

        Parameters
        ----------
        rhs : np.ndarray
            Right hand as given above.
        t : float
            Current timestamp.
            Future one will be computed as `t + dt`
        dt : float
            Time step size.
        """
        raise Exception("TODO: `fSolve2` not implemented for this DE solver.")

    @abstractmethod
    def initial_u0(self, mode: str) -> np.ndarray:
        """Compute some initial conditions for the equation.

        Parameters
        ----------
        mode : str
            The type of initial condition to generate.
        """
        pass

    @abstractmethod
    def u_solution(self, u0: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the (analytical) solution at time `t`.

        Parameters
        ----------
        u0 : np.ndarray
            Array of shape (N,) representing the initial condition.
        t : float
            Time at which to evaluate the solution.
        """
        pass
