import pytest
import numpy as np

from qmat.qcoeff import Q_GENERATORS, genQCoeffs
from qmat.utils import getClasses

GENERATORS = getClasses(Q_GENERATORS)


@pytest.mark.parametrize("name", GENERATORS.keys())
def testGeneration(name):
    gen = GENERATORS[name].getInstance()
    n1, w1, Q1 = gen.nodes, gen.weights, gen.Q

    assert type(n1) == np.ndarray, \
        f"nodes for {name} are not np.ndarray but {type(n1)}"
    assert type(w1) == np.ndarray, \
        f"weights for {name} are not np.ndarray but {type(w1)}"
    assert type(Q1) == np.ndarray, \
        f"Q for {name} are not np.ndarray but {type(Q1)}"

    assert n1.ndim == 1, f"nodes for {name} are not 1D : {n1}"
    assert w1.ndim == 1, f"weights for {name} are not 1D : {n1}"
    assert n1.size == w1.size, \
        f"nodes and weights for {name} don't have the same size : {n1} / {w1}"
    assert Q1.shape == (n1.size, n1.size), \
        f"Q for {name} has inconsistent shape : {Q1.shape}"
    assert gen.nNodes == n1.size, \
        f"nNodes property from {name} is not equal to node size !"

    try:
        n2, w2, Q2 = genQCoeffs(name)
    except TypeError:
        n2, w2, Q2 = genQCoeffs(name, **GENERATORS[name].DEFAULT_PARAMS)
    assert np.allclose(n1, n2), \
        f"OOP nodes {n1} and PP nodes {n2} are not equals for {name}"
    assert np.allclose(w1, w2), \
        f"OOP weights {w1} and PP weights {w2} are not equals for {name}"
    assert np.allclose(Q1, Q2), \
        f"OOP Q matrix {Q1} and PP Q matrix {Q2} are not equals for {name}"


@pytest.mark.parametrize("name", GENERATORS.keys())
def testAdditionalCoeffs(name):
    gen = GENERATORS[name].getInstance()
    nodes, S1, h1 = gen.nodes, gen.S, gen.hCoeffs

    assert type(S1) == np.ndarray, \
        f"S for {name} are not np.ndarray but {type(S1)}"
    assert type(h1) == np.ndarray, \
        f"hCoeffs for {name} are not np.ndarray but {type(h1)}"

    assert S1.shape == (nodes.size, nodes.size), \
        f"S for {name} has inconsistent shape : {S1.shape}"
    assert h1.ndim == 1, f"hCoeffs for {name} are not 1D : {h1}"
    assert h1.size == nodes.size, \
        f"hCoeffs for {name} has inconsistent size : {h1.size}"

    try:
        _, _, _, S2, h2 = genQCoeffs(name, withS=True, hCoeffs=True)
    except TypeError:
        _, _, _, S2, h2 = genQCoeffs(name, withS=True, hCoeffs=True,
                                **GENERATORS[name].DEFAULT_PARAMS)
    assert np.allclose(S1, S2), \
        f"OOP S matrix {S1} and PP S matrix {S2} are not equals for {name}"
    assert np.allclose(h1, h2), \
        f"OOP hCoeffs {h1} and PP hCoeffs {h2} are not equals for {name}"


    try:
        try:
            _, b, _  = genQCoeffs(name, embedded=True)
        except TypeError:
            _, b, _  = genQCoeffs(name, embedded=True, **GENERATORS[name].DEFAULT_PARAMS)

        assert type(b) == np.ndarray
        assert b.ndim == 2
    except NotImplementedError:
        pass


@pytest.mark.parametrize("name", GENERATORS.keys())
def testS(name):
    gen = GENERATORS[name].getInstance()
    Q, S, T, Tinv = gen.Q, gen.S, gen.T, gen.Tinv

    Q2S = T @ Q
    assert np.allclose(Q2S, S), f"Q transformed to S is not equal to S for {name}"

    Q2S2Q = Tinv @ Q2S
    assert np.allclose(Q2S2Q, Q), \
        f"Q transformed to S then back to Q is not equal to Q for {name}"
