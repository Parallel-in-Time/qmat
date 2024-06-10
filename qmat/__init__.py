#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation of Q-coefficients for Spectral Deferred Corrections (and other time-integration methods ...)
"""
__version__ = "0.0.1"

from qmat.qcoeff import genQCoeffs, Q_GENERATORS
from qmat.qdelta import genQDeltaCoeffs, QDELTA_GENERATORS

__all__ = [
    "genQCoeffs", "genQDeltaCoeffs",
    "Q_GENERATORS", "QDELTA_GENERATORS"]
