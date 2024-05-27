import pytest
import numpy as np

from qmat.nodes import NodesGenerator, NODE_TYPES

def chebyNodes(kind, n):
    i = np.arange(n, dtype=float) + 1
    i = i[-1::-1]
    if kind == 1:
        nodes = np.cos((i - 0.5) / n * np.pi)
    elif kind == 2:
        nodes = np.cos(i / (n + 1) * np.pi)
    elif kind == 3:
        nodes = np.cos((i - 0.5) / (n + 0.5) * np.pi)
    elif kind == 4:
        nodes = np.cos(i / (n + 0.5) * np.pi)
    return tuple(nodes)

def equidNodes(n):
    return tuple(np.linspace(-1, 1, num=n + 2)[1:-1])

REF_NODES = {
    'LEGENDRE': {
        2: (-1 / 3**0.5, 1 / 3**0.5),
        3: (-((3 / 5) ** 0.5), 0, (3 / 5) ** 0.5),
        4: (
            -((3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            -((3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            (3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5,
            (3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5,
        ),
        5: (
            -1 / 3 * (5 + 2 * (10 / 7) ** 0.5) ** 0.5,
            -1 / 3 * (5 - 2 * (10 / 7) ** 0.5) ** 0.5,
            0,
            1 / 3 * (5 - 2 * (10 / 7) ** 0.5) ** 0.5,
            1 / 3 * (5 + 2 * (10 / 7) ** 0.5) ** 0.5,
        ),
    }
}
nTests = list(REF_NODES['LEGENDRE'].keys())
for kind in [1, 2, 3, 4]:
    REF_NODES[f'CHEBY-{kind}'] = {n: chebyNodes(kind, n) for n in nTests}
REF_NODES['EQUID'] = {n: equidNodes(n) for n in nTests}


@pytest.mark.parametrize("nodeType", REF_NODES.keys())
def testGauss(nodeType):
    gen = NodesGenerator(nodeType=nodeType, quadType='GAUSS')
    ref = REF_NODES[nodeType]
    for n, nodes in ref.items():
        assert np.allclose(nodes, gen.getNodes(n)), f"difference with reference nodes (n={n})"


@pytest.mark.parametrize("quadType", ["LOBATTO", "RADAU-RIGHT", "RADAU-LEFT"])
@pytest.mark.parametrize("nodeType", NODE_TYPES)
def testNonGauss(nodeType, quadType):
    gen = NodesGenerator(nodeType=nodeType, quadType=quadType)
    for n in nTests:
        nodes = gen.getNodes(n)
        if quadType in ["LOBATTO", "RADAU-RIGHT"]:
            assert nodes[-1] == 1, f"right node not equal to 1 (n={n})"
        if quadType in ["LOBATTO", "RADAU-LEFT"]:
            assert nodes[0] == -1, f"left node not equal to -1 (n={n})"
        assert np.allclose(np.sort(nodes), nodes), f"nodes not ordered (n={n})"


@pytest.mark.parametrize("quadType", ["GAUSS", "LOBATTO", "RADAU-RIGHT", "RADAU-LEFT"])
@pytest.mark.parametrize("nodeType", ["LEGENDRE", "CHEBY-1", "CHEBY-2", "CHEBY-3", "CHEBY-4"])
def testAsymptotic(nodeType, quadType):
    gen = NodesGenerator(nodeType=nodeType, quadType=quadType)
    unif = NodesGenerator(nodeType="EQUID", quadType=quadType)
    n = 1000
    nodes = gen.getNodes(n)
    s = (unif.getNodes(n) + 1)/2
    limit = -np.cos(s*np.pi)
    assert np.max(np.abs(limit-nodes)) < 0.01
