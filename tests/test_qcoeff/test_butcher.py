import pytest
import numpy as np

from qmat.qcoeff.butcher import RK_SCHEMES

@pytest.mark.parametrize("lam", [-1, 1j, -0.5+0.5j])
@pytest.mark.parametrize("scheme", RK_SCHEMES.keys())
@pytest.mark.parametrize("which", ["LEFT", "RIGHT", "BOTH"])
def testPadding(which, scheme, lam):
    ref = RK_SCHEMES[scheme]().solveDahlquist(lam, 1, 1, 1)
    pad = RK_SCHEMES[scheme](padding=which).solveDahlquist(lam, 1, 1, 1)
    assert np.allclose(ref, pad), f"{which} padded Butcher table produces inconsistent result for {scheme}"
