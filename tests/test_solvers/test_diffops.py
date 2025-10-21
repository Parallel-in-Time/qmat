import pytest

from qmat.solvers.generic.diffops import DIFFOPS

@pytest.mark.parametrize("name", DIFFOPS.keys())
def testDiffOps(name):
    DIFFOPS[name].test()