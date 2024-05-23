#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the qmat package
"""
from qmat.qcoeff import Q_GENERATORS
from qmat.qdelta import QDELTA_GENERATORS

def genQCoeffs(qType, **params):
    try:
        Generator = Q_GENERATORS[qType]
    except KeyError:
        raise ValueError(f"qType {qType} not available")
    gen = Generator(**params)
    return gen.nodes, gen.weights, gen.Q


def genQDeltaCoeffs(qDeltaType, nodes, Q, nIter=None, **params):
    pass # TODO : implement variable sweeps ...
