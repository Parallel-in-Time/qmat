import pytest

from qmat.solvers.generic.diffops import DIFFOPS, DiffOp


def testBase():
    diffOpSmall = DiffOp(10*[0.0])
    diffOpLarge = DiffOp(1000*[0.0])
    assert diffOpSmall.innerSolver != diffOpLarge.innerSolver


@pytest.mark.parametrize("name", DIFFOPS.keys())
def testImplementations(name):
    DIFFOPS[name].test()
