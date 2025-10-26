import numpy as np

from qmat.solvers.dahlquist import DahlquistIMEX
from qmat.solvers.generic import DiffOp, CoeffSolver


class DiffOpIMEX(DiffOp):
    """
    Base class for an IMEX differential operator
    """

    def evalF2(self, u:np.ndarray, t:float, out:np.ndarray):
        """
        Evaluate f_EX(u,t) and store the result into out
        """
        raise NotImplementedError("evalF must be provided")


class CoeffSolverIMEX(CoeffSolver):
    """
    Coefficient based solver class for IMEX differential operators.
    """
    def __init__(self, diffOp, tEnd=1, nSteps=1, t0=0):
        self.diffOp: DiffOpIMEX = None
        assert isinstance(diffOp, DiffOpIMEX), \
            f"DiffOpIMEX object is required for diffOp argument, not {diffOp}"
        super().__init__(diffOp, tEnd, nSteps, t0)


    def evalF2(self, u:np.ndarray, t:float, out:np.ndarray):
        self.diffOp.evalF2(u, t, out)

    def solve(self, QI, wI, QE, wE, uNum=None):
        nNodes, QI, wI, QE, wE, useWeights = DahlquistIMEX.checkCoeff(QI, wI, QE, wE)

        assert self.lowerTri(QI), \
            "lower triangular matrix QI expected for non-linear IMEX solver"
        assert self.lowerTri(QE, strict=True), \
            "strictly lower triangular matrix QE expected for non-linear IMEX solver"
        QI, QE = self.dt*QI, self.dt*QE
        if useWeights:
            wI, wE = self.dt*wI, self.dt*wE

        if uNum is None:
            uNum = np.zeros((self.nSteps+1, *self.uShape), dtype=self.dtype)
            uNum[0] = self.u0

        rhs = np.zeros(self.uShape, dtype=self.dtype)
        fEvals = np.zeros((nNodes, *self.uShape), dtype=self.dtype)
        fEvals2 = np.zeros((nNodes, *self.uShape), dtype=self.dtype)

        times = np.linspace(self.t0, self.tEnd, self.nSteps+1)
        tau = QI.sum(axis=1)

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
                    self.axpy(a=QI[m, j], x=fEvals[j], y=rhs)
                    self.axpy(a=QE[m, j], x=fEvals2[j], y=rhs)

                # solve node (if non-zero diagonal coefficient)
                if QI[m, m] != 0:
                    self.fSolve(a=QI[m, m], rhs=rhs, t=tNode, out=uNode)
                else:
                    np.copyto(uNode, rhs)

                # evalF on current stage
                self.evalF(u=uNode, t=tNode, out=fEvals[m])
                self.evalF2(u=uNode, t=tNode, out=fEvals2[m])

            # step update (if not, uNum[i+1] is already the last stage)
            if useWeights:
                np.copyto(uNum[i+1], uNum[i])
                for m in range(nNodes):
                    self.axpy(a=wI[m], x=fEvals[m], y=uNum[i+1])
                    self.axpy(a=wE[m], x=fEvals2[m], y=uNum[i+1])

        return uNum