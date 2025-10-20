import numpy as np
from qmat.playground.diff_eqs.de_solver import DESolver


class Dahlquist(DESolver):

    def __init__(self, lam1: float, lam2: float):
        # Lambda 1
        self.lam1: float = lam1

        # Lambda 2
        self.lam2: float = lam2
