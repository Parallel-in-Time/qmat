import pytest
import numpy as np
from physverify.butcher import check_tableau as checkButcher

from qmat.qcoeff.butcher import RK_SCHEMES, RK

@pytest.mark.parametrize("lam", [-1, 1j, -0.5+0.5j])
@pytest.mark.parametrize("scheme", RK_SCHEMES.keys())
@pytest.mark.parametrize("which", ["LEFT", "RIGHT", "BOTH"])
def testPadding(which, scheme, lam):
    ref:RK = RK_SCHEMES[scheme]().solveDahlquist(lam, 1, 1, 1)
    pad:RK = RK_SCHEMES[scheme](padding=which).solveDahlquist(lam, 1, 1, 1)
    assert np.allclose(ref, pad), f"{which} padded Butcher table produces inconsistent result for {scheme}"


@pytest.mark.parametrize("scheme", RK_SCHEMES.keys())
def testOrderConditions(scheme):
    gen:RK = RK_SCHEMES[scheme]()
    check = checkButcher(
        gen.A, gen.b, gen.c, claimed=gen.order, claimed_linear=gen.orderLinear)
    assert check.ok, check.message
