#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for Q coefficients generation
"""
import pkgutil
from qmat.utils import checkOverriding, storeClass
from qmat.lagrange import LagrangeApproximation

class QGenerator(object):

    @property
    def nodes(self):
        raise NotImplementedError("mouahahah")

    @property
    def Q(self):
        raise NotImplementedError("mouahahah")

    @property
    def weights(self):
        raise NotImplementedError("mouahahah")

    @property
    def S(self):
        pass

    @property
    def hCoeffs(self):
        approx = LagrangeApproximation(self.nodes)
        return approx.getInterpolationMatrix([1]).ravel()


Q_GENERATORS:dict[str:QGenerator] = {}

def register(cls:QGenerator):
    for name in ["nodes", "Q", "weights"]:
        checkOverriding(cls, name)
    storeClass(cls, Q_GENERATORS)
    return cls

# Work the magic !
__all__ = [name for name in locals().keys() if not name.startswith('__')]
for _, moduleName, _ in pkgutil.walk_packages(__path__):
    __all__.append(moduleName)
    __import__(__name__+'.'+moduleName)
