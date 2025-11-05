#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of time-integration solvers that make use of `qmat`-generated coefficients.

    🔔 Those are not fully optimized implementations of their corresponding
    time-integration scheme, but conveniences classes allowing
    some first **experiments** with your problem(s) of interest.
    They are mostly given

**Modules** ⚙️

- :class:`sdc` : functions to run SDC on a scalar Dahlquist problem and evaluate its numerical error or convergence order
- :class:`dahlquist` : generic coefficient-based time-integration solver for the (IMEX) vectorized Dahlquist problem

**Sub-package** 📦

- :class:`generic` : time-integration solvers for generic (systems of / non-linear) ODEs
"""
