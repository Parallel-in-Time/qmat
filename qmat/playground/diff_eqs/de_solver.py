from abc import ABC, abstractmethod
import numpy as np


class DESolver(ABC):

    @abstractmethod
    def eval_f(self, u: np.ndarray) -> np.ndarray:
        """Evaluate the right-hand side of the 1D viscous Burgers' equation.

        Parameters
        ----------
        u : np.ndarray
            Array of shape (N,) representing the solution at the current time step.

        Returns
        -------
        f : np.ndarray
            Array of shape (N,) representing the right-hand side evaluated at `u`.
        """
