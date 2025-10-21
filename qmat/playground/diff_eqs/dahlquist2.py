import numpy as np
from qmat.playground.diff_eqs.de_solver import DESolver


class Dahlquist2(DESolver):
    """
    Modified Dahlquist test equation with superposition of two
    frequencies in the solution u(t).

    u(t) = 0.5*(exp(lam1*t) + exp(lam2*t)) * u(0)
    """

    def __init__(self, lam1: complex, lam2: complex):
        # Lambda 1
        self.lam1: float = lam1

        # Lambda 2
        self.lam2: float = lam2

    def initial_u0(self, mode: str = None) -> np.ndarray:
        return np.array([1.0 + 0.0j])

    def du_dt(self, u: np.ndarray, t: float) -> np.ndarray:
        retval = (
            (self.lam1 * np.exp(t * self.lam1) + self.lam2 * np.exp(t * self.lam2))
            / (np.exp(t * self.lam1) + np.exp(t * self.lam2))
            * u
        )
        assert retval.shape == u.shape
        return retval

    def u_solution(self, u0: np.ndarray, t: float) -> np.ndarray:
        assert isinstance(t, float)
        retval = 0.5*(np.exp(t * self.lam1) + np.exp(t * self.lam2)) * u0
        assert retval.shape == u0.shape
        return retval
