import numpy as np
from qmat.playgrounds.martin.diff_eqs.de_solver import DESolver


class TwoFreq(DESolver):
    """
    Test equation using a matrix to generate a
    superposition of two frequencies in the solution.
    """

    def __init__(self, lam1: complex, lam2: complex, lam3: complex = None, lam4: complex = None):
        """
        :param lam1: L[0,0]
        :param lam2: L[0,1]
        :param lam3: L[1,0]
        :param lam4: L[1,1]
        """

        self.lam1: complex = lam1
        self.lam2: complex = lam2
        self.lam3: complex = lam3 if lam3 is not None else 0
        self.lam4: complex = lam4 if lam4 is not None else lam2

        self.L = np.array([[self.lam1, self.lam2], [self.lam3, self.lam4]], dtype=np.complex128)

        # compute eigenvalues and eigenvectors of L
        self.eigvals, self.eigvecs = np.linalg.eig(self.L)

        if not np.all(np.isclose(np.real(self.eigvals), 0)):
            raise Exception("Dahlquist3 matrix L must have purely imaginary eigenvalues")

    def initial_u0(self, mode: str = None) -> np.ndarray:
        return np.array([1, 0.1], dtype=np.complex128)

    def du_dt(self, u: np.ndarray, t: float) -> np.ndarray:
        assert isinstance(t, float)
        retval = self.L @ u
        assert retval.shape == u.shape
        return retval

    def u_solution(self, u0: np.ndarray, t: float) -> np.ndarray:
        from scipy.linalg import expm

        assert isinstance(t, float)
        retval = expm(self.L * t) @ u0
        assert retval.shape == u0.shape
        return retval
