import numpy as np
from qmat.playgrounds.martin.diff_eqs.de_solver import DESolver


class Dahlquist2(DESolver):
    """
    Modified Dahlquist test equation with superposition of two
    frequencies in the solution u(t).

    u(t) = 0.5*(exp(lam1*t) + exp(lam2*t)) * u(0)
    """

    def __init__(self, lam1: complex, lam2: complex, s: float = 0.6):
        """
        :param lam1: First eigenvalue (complex)
        :param lam2: Second eigenvalue (complex)
        :param s: Weighting between the two exponentials in the solution
        """
        self.lam1: complex = lam1
        self.lam2: complex = lam2
        # Weighting between the two exponentials in the solution
        # to avoid division by 0
        self.s: float = s

        assert 0 <= self.s <= 1, "s must be in [0,1]"

    def initial_u0(self, mode: str = None) -> np.ndarray:
        return np.array([1.0 + 0.0j], dtype=np.complex128)

    def du_dt(self, u: np.ndarray, t: float) -> np.ndarray:
        retval = (
            (self.lam1 * self.s * np.exp(t * self.lam1) + self.lam2 * (1.0 - self.s) * np.exp(t * self.lam2))
            / (self.s * np.exp(t * self.lam1) + (1.0 - self.s) * np.exp(t * self.lam2))
            * u
        )
        assert retval.shape == u.shape
        return retval

    def u_solution(self, u0: np.ndarray, t: float) -> np.ndarray:
        assert isinstance(t, float)
        retval = (self.s * np.exp(t * self.lam1) + (1.0 - self.s) * np.exp(t * self.lam2)) * u0
        assert retval.shape == u0.shape
        return retval
