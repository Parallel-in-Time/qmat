#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for QDelta coefficients generation
"""
import pkgutil
import numpy as np

from qmat.utils import checkOverriding, storeClass

class QDeltaGenerator(object):

    def __init__(self, nodes, Q, **params):
        self.nodes = np.asarray(nodes).ravel()
        self.Q = Q
        self.params = params

        nNodes = self.nNodes
        self.QDelta = np.zeros((nNodes, nNodes), dtype=float)

    @property
    def nNodes(self):
        return self.nodes.size

    def getQDelta(self, k=None):
        raise NotImplementedError("mouahahah")

    @property
    def dTau(self):
        return self.nodes*0

QDELTA_GENERATORS:dict[str:QDeltaGenerator] = {}

def register(cls:QDeltaGenerator):
    for name in ["getQDelta"]:
        checkOverriding(cls, name, isProperty=False)
    storeClass(cls, QDELTA_GENERATORS)
    return cls

# Work the magic !
__all__ = [name for name in locals().keys() if not name.startswith('__')]
for _, moduleName, _ in pkgutil.walk_packages(__path__):
    __all__.append(moduleName)
    __import__(__name__+'.'+moduleName)
