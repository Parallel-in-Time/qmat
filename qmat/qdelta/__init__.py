#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Defines the base abstract class to generate  :math:`Q_\Delta` approximations :
the :class:`QDeltaGenerator` 🚀

Each submodule contains specializations of this class for many kind of
methods :

- :class:`timestepping` : based on time-stepping methods (Backward Euler, etc ...)
- :class:`algebraic` : based on algebraic consideration on the :math:`Q` matrix
- :class:`min` : diagonal approximations based on minimization

Examples
--------

>>> qGen:QGenerator = ... # any QGenerator object implemented in qmat.qcoeff.[...]
>>>
>>> # Generate QDelta coefficients with generic function
>>> from qmat.qdelta import genQDeltaCoeffs
>>> qDeltaBE = genQDeltaCoeffs("BE", nodes=qGen.nodes)  # Backward Euler approximation
>>> qDeltaLU = genQDeltaCoeffs("LU", Q=qGen.Q)          # LU approximation
>>>
>>> # Generate QDelta coefficients with QDeltaGenerator objects
>>> from qmat.qdelta.timestepping import BE
>>> qDeltaBE = BE(nodes=qGen.nodes).getQDelta()
>>> from qmat.qdelta.algebraic import LU
>>> qDeltaLU = LU(Q=qGen.Q)
>>>
>>> # Simplified import for all QDeltaGenerator objects
>>> from qmat.qdelta import QDELTA_GENERATORS   # 💡 can also be imported from qmat directly
>>> qDeltaBE = QDELTA_GENERATORS["BE"](nodes=qGen.nodes).getQDelta()
>>> qDeltaLU = QDELTA_GENERATORS["LU"](Q=qGen.Q).getQDelta()

Note
----
📣 If you want to **cover all available approximations** implemented in `qmat`,
we highly suggest to use the `qGen` keyword argument, allowing to extract any
required parameter from a `QGenerator` object, e.g :

>>> # Using generic function
>>> for qdType in ["BE", "LU", "MIN-SR-S", "MIN-SR-NS", "MIN-SR-FLEX"]:
>>>     qDelta = genQDeltaCoeffs(qdType, qGen=qGen)
>>>
>>> # Using QDeltaGenerator objects
>>> for qdType in ["BE", "LU", "MIN-SR-S", "MIN-SR-NS", "MIN-SR-FLEX"]:
>>>     qDelta = QDELTA_GENERATORS[qdType](qGen=qGen).getQDelta()

💡 This ensure forward compatibility in your code, so it can use any other
:math:`Q_\Delta` approximations added later in `qmat` without modification.

    ⚠️ You can also specify :math:`Q_\Delta` specific parameter(s) in addition to `qGen` :
    in that case, the corresponding parameters extracted from `qGen` are **overridden**.
"""
import inspect
import numpy as np
from typing import TypeVar

from qmat.utils import checkOverriding, storeClass, importAll, checkGenericConstr, useQGen
from qmat.qcoeff import QGenerator

T = TypeVar("T")


class QDeltaGenerator(object):
    r"""
    Base abstract class for :math:`Q_\Delta` coefficients generators.

    Parameters
    ----------
    Q : np.ndarray
        The :math:`Q` matrix of the base approximated method.
    **kwargs :
        Additional parameters given in for generic calls, ignored by the class.
    """

    _K_DEPENDENT = False

    def isKDependent(cls):
        r"""Wether or not the :math:`Q_\Delta` coefficients varies with the iterations"""
        return cls._K_DEPENDENT

    def __init__(self, Q, **kwargs):
        self.Q:np.ndarray = np.asarray(Q, dtype=float)
        """:math:`Q` matrix of the approximated time-integration method"""

    @staticmethod
    def extractParams(qGen:QGenerator) -> dict:
        r"""
        Extract from a :math:`Q`-generator object all parameters
        required to instantiate the :math:`Q_\Delta`-generator
        """
        assert isinstance(qGen, QGenerator), "qGen parameter must be a QGenerator object"
        return {"Q": qGen.Q}

    @property
    def size(self)->int:
        """Dimension of the approximated :math:`Q`-coefficients (number of nodes)"""
        return self.Q.shape[0]

    @property
    def zeros(self)->np.ndarray:
        """Seros matrix with the same size of the underlying :math:`Q` matrix"""
        M = self.size
        return np.zeros((M, M), dtype=float)

    def computeQDelta(self, k=None) -> np.ndarray:
        r"""
        Compute and returns the :math:`Q_\Delta` matrix, has to be implemented
        in the specialized class.

        Parameters
        ----------
        k : int, optional
            Iteration number of the approximation. The default is None.

        Returns
        -------
        QDelta : np.ndarray
        """
        raise NotImplementedError(f"abstract class {type(self).__name__} should not be used")

    def getQDelta(self, k=None, copy=True):
        r"""
        Generic method to retrieve the :math:`Q_\Delta` coefficients

        Parameters
        ----------
        k : int, optional
            Iteration number of the approximation (if needed). The default is None.
        copy : bool, optional
            Return a copy of the the result returned by `computeQDelta`.
            The default is True.

        Returns
        -------
        QDelta : np.ndarray
        """
        try:
            QDelta = self._QDelta[k] if self._K_DEPENDENT else self._QDelta
        except Exception as e:
            QDelta = self.computeQDelta(k)
            if type(e) == AttributeError:
                self._QDelta = {k: QDelta} if self._K_DEPENDENT else QDelta
            elif type(e) == KeyError:
                self._QDelta[k] = QDelta
            else:
                raise Exception("some very weird bug happened ... did you do fishy stuff ?")
        return QDelta.copy() if copy else QDelta

    def getSDelta(self, k=None):
        r"""
        Compute the :math:`S_\Delta` matrix (approximation of :math:`S`).

        Parameters
        ----------
        k : int, optional
            Iteration number, used when the approximation depends on it.
            The default is None.

        Returns
        -------
        SDelta : np.ndarray
        """
        QDelta = self.getQDelta(k)
        M = QDelta.shape[0]
        T = np.eye(M)
        T[1:,:-1][np.diag_indices(M-1)] = -1
        return T @ QDelta

    @property
    def dTau(self)->np.ndarray:
        r"""The :math:`\delta_\tau` coefficients associated to :math:`Q_\Delta`"""
        return np.zeros(self.size, dtype=float)

    def genCoeffs(self, k=None, form="Z2N", dTau=False):
        r"""
        Generic method to produce :math:`Q_\Delta` coefficients

        Parameters
        ----------
        k : int or list, optional
            Iteration(s) for the approximation. The default is None.
        form : str, optional
            Build approximation in zero-to-nodes (Z2N) or node-to-node (N2N).
            The default is "Z2N".
        dTau : bool, optional
            Wether or not to return the :math:`\delta_\tau`.
            The default is False.

        Returns
        -------
        np.ndarray or tuple
            If `k` is a scalar or `None`, returns a MxM matrix.
            If `k` is a list, returns a len(k)xMxM matrix.
            If `dTau=True`, returns a tuple `(QDelta, dTau)`.
        """
        if form == "Z2N":
            gen = lambda k, copy=False: self.getQDelta(k, copy)
        elif form == "N2N":
            gen = lambda k, copy=None: self.getSDelta(k)
        else:
            raise ValueError(f"form must be Z2N or N2N, not {form}")
        try:
            k = list(k)
            out = [np.array([gen(_k, copy=False) for _k in k])]
        except TypeError:
            out = [gen(k)]
        if dTau:
            out += [self.dTau]
        return out if len(out) > 1 else out[0]


QDELTA_GENERATORS: dict[str, type[QDeltaGenerator]] = {}
"""Dictionary containing all specialized :class:`QDeltaGenerator` classes"""

def register(cls: type[T]) -> type[T]:
    """Class decorator to register a specialized :class:`QDeltaGenerator` class in `qmat`"""
    checkGenericConstr(cls)
    checkOverriding(cls, "computeQDelta", isProperty=False)
    try:
        sig = inspect.signature(cls.computeQDelta)
        par = sig.parameters["k"]
        assert par.kind == par.POSITIONAL_OR_KEYWORD
        if par.default is not None:
            cls._K_DEPENDENT = True
    except (KeyError, AssertionError):
        raise AssertionError(f"{cls.__name__} class does not properly override the computeQDelta method")
    cls.__init__ = useQGen(cls.__init__)
    storeClass(cls, QDELTA_GENERATORS)
    return cls


def genQDeltaCoeffs(qDeltaType, nSweeps=None, form="Z2N", dTau=False, **params):
    r"""
    Generic function to produce :math:`Q_\Delta` coefficients

    Parameters
    ----------
    qDeltaType : str or list
        The type of approximation, can be a list to have several sweeps.
    nSweeps : int, optional
        Number of sweeps when :math:`Q_\Delta` matrices are required for
        several sweeps. The default is None.
    form : str, optional
        Build approximation in zero-to-nodes (Z2N) or node-to-node (N2N).
        The default is "Z2N".
    dTau : bool, optional
        Wether or not to return the :math:`\delta_\tau`. The default is False.
    **params
        Additional arguments used to instantiate all :class:`QDeltaGenerator`

    Returns
    -------
    np.ndarray or tuple
        If `qDeltaType` is a string, returns a :math:`M \times M` matrix.
        If `qDeltaType` is a list or `nSweeps != None`,
        returns a :math:`N_{sweeps} \times M \times M` matrix.
        If `dTau=True`, returns a tuple `(QDelta, dTau)`.
    """
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
                f"nSweeps ({nSweeps}) is lower than list length for qDeltaType ({qDeltaType})"
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
        return gen.genCoeffs(form=form, dTau=dTau)

    else:  # Multiple matrices return
        try:
            Generators = [QDELTA_GENERATORS[qDT] for qDT in qDeltaType]
        except KeyError:
            raise ValueError(f"qDeltaType={qDeltaType} is not available")

        if len(qDeltaType) == 1:  # Single QDelta generator
            gen = Generators[0](**params)
            return gen.genCoeffs(
                k=[k+1 for k in range(nSweeps)], form=form, dTau=dTau)

        else:  # Multiple QDelta generators
            gens = [Gen(**params) for Gen in Generators]
            out = [np.array(
                [gen.genCoeffs(k+1, form) for k, gen in enumerate(gens)]
                )]
            if dTau:
                out += [gens[0].dTau]

    return out if len(out) > 1 else out[0]


if __name__ != "__main__":
    # Import all local submodules
    __all__ = ["genQDeltaCoeffs", "QDeltaGenerator", "QDELTA_GENERATORS", "register"]
    importAll(locals(), __all__, __path__, __name__, __import__)
