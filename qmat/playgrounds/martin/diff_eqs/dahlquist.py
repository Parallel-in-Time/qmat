import numpy as np
from qmat.playgrounds.martin.diff_eqs.de_solver import DESolver


class Dahlquist(DESolver):
    """
    Standard Dahlquist test equation with two eigenvalues.
    Optionally, some external frequency forcing can be added which is
    configurable through `ext_scalar`.

    u(t) = exp(t*(lam1+lam2))*u(0) + s*sin(t)

    d/dt u(t) = (lam1 + lam2) * (u(t) - s*sin(t)) + s*cos(t)
    """

    def __init__(self, lam1: complex, lam2: complex, ext_scalar: float = 0.0):
        # Lambda 1
        self.lam1: float = lam1

        # Lambda 2
        self.lam2: float = lam2

        # Scaling of external sin(t) frequency. Set to 0 to deactivate.
        self.ext_scalar = ext_scalar

    def initial_u0(self, mode: str = None) -> np.ndarray:
        return np.array([1.0 + 0.0j], dtype=np.complex128)

    def evalF(self, u: np.ndarray, t: float) -> np.ndarray:
        lam = self.lam1 + self.lam2
        s = self.ext_scalar
        retval = lam * (u - s * np.sin(t)) + s * np.cos(t)

        assert retval.shape == u.shape
        return retval

    def fSolve(self, u: np.ndarray, dt: float, t: float) -> np.ndarray:
        t1 = t + dt
        lam = self.lam1 + self.lam2
        s = self.ext_scalar

        rhs = u - s * dt * (lam * np.sin(t1) - np.cos(t1))
        retval = rhs / (1.0 - dt * lam)

        assert retval.shape == u.shape
        return retval

    def u_solution(self, u0: np.ndarray, t: float) -> np.ndarray:
        lam = self.lam1 + self.lam2
        s = self.ext_scalar
        assert isinstance(t, float)
        retval = np.exp(t * lam) * u0 + s * np.sin(t)

        assert retval.shape == u0.shape
        return retval
