#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for QDelta coefficients generation
"""
import numpy as np

from qmat.utils import checkOverriding, storeClass, importAll, checkGenericConstr

class QDeltaGenerator(object):

    def __init__(self, Q, **kwargs):
        self.Q = np.asarray(Q, dtype=float)
        self.QDelta = np.zeros_like(self.Q)

    def storeAndReturn(self, QDelta):
        np.copyto(self.QDelta, QDelta)
        return self.QDelta

    def getQDelta(self, k=None):
        raise NotImplementedError("mouahahah")

    @property
    def dTau(self):
        return self.QDelta[0]*0

    def genCoeffs(self, k=None, dTau=False):
        if isinstance(k, list):
            out = [np.array([self.getQDelta(_k) for _k in k])]
        else:
            out = [self.getQDelta(k)]
        if dTau:
            out += [self.dTau]
        return out if len(out) > 1 else out[0]

QDELTA_GENERATORS = {}

def register(cls:QDeltaGenerator)->QDeltaGenerator:
    checkGenericConstr(cls)
    checkOverriding(cls, "getQDelta", isProperty=False)
    storeClass(cls, QDELTA_GENERATORS)
    return cls

def genQDeltaCoeffs(qDeltaType, nSweeps=None, dTau=False, **params):

    # Check arguments
    if isinstance(qDeltaType, str):
        if nSweeps is None:
            pass  # only one QDelta matrix, default approach
        elif isinstance(nSweeps, int) and nSweeps > 0:
            qDeltaType = [qDeltaType]  # more sweeps of the same QDelta matrix
        else:
            raise ValueError(f"bad value of nSweep {nSweeps}")
    elif isinstance(qDeltaType, list):
        assert len(qDeltaType) > 0, "need at least one qDeltaType in the list"
        if nSweeps is None:
            nSweeps = len(qDeltaType)  # number of sweeps given in the list
        elif isinstance(nSweeps, int) and nSweeps > 0:
            # complete with additional sweeps
            assert nSweeps >= len(qDeltaType), \
                f"nSweeps ({nSweeps} lower than list length for qDeltaType ({qDeltaType})"
            qDeltaType += [qDeltaType[-1]]*(nSweeps-len(qDeltaType))
        else:
            raise ValueError(f"bad value of nSweep {nSweeps}")
    else:
        raise ValueError(f"bad value of qDeltaType {qDeltaType}")

    if nSweeps is None:  # Single matrix return

        try:
            Generator = QDELTA_GENERATORS[qDeltaType]
        except KeyError:
            raise ValueError(f"qDeltaType={qDeltaType} is not available")

        gen = Generator(**params)
        return gen.genCoeffs(dTau=dTau)

    else:  # Multiple matrices return
        try:
            Generators = [QDELTA_GENERATORS[qDT] for qDT in qDeltaType]
        except KeyError:
            raise ValueError(f"qDeltaType={qDeltaType} is not available")

        if len(qDeltaType) == 1:  # Single QDelta generator
            gen = Generators[0](**params)
            return gen.genCoeffs(k=[k+1 for k in range(nSweeps)], dTau=dTau)

        else:  # Multiple QDelta generators
            gens = [Gen(**params) for Gen in Generators]
            out = [np.array(
                [gen.getQDelta(k+1) for k, gen in enumerate(gens)]
                )]
            if dTau:
                out += [gens[0].dTau]

    return out if len(out) > 1 else out[0]


# Import all local submodules
__all__ = importAll(locals(), __path__, __name__, __import__)
