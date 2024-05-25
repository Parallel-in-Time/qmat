#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility module
"""
import pkgutil
import numpy as np

def checkOverriding(cls, name, isProperty=True):
    method = getattr(cls, name)
    parent = getattr(cls.mro()[-2], name)
    assert method != parent, \
        f"{name} method must be overriden in {cls.__name__} class"
    if isProperty:
        assert type(method) == property, \
            f"{name} method must be a property in {cls.__name__} class"
    else:
        pass
        # TODO : check that signatures are the same


def storeAlias(cls, dico, alias):
    assert alias not in dico, f"{alias} alias already registered in {dico}"
    dico[alias] = cls


def storeClass(cls, dico):
    storeAlias(cls, dico, cls.__name__)
    aliases = getattr(cls, "aliases", None)
    if aliases:
        assert isinstance(aliases, list), \
            f"aliases must be a list in class {cls.__name__}"
        for alias in aliases:
            storeAlias(cls, dico, alias)


def importAll(localVars, path, name, _import):
    """The magic function"""
    _all = [var for var in localVars.keys() if not var.startswith('__')]
    for _, moduleName, _ in pkgutil.walk_packages(path):
        _all.append(moduleName)
        _import(name+'.'+moduleName)


def getClasses(dico, module=None):
    classes = {}
    if module is None:
        check = lambda cls: True
    else:
        check = lambda cls: cls.__module__.endswith("."+module)
    for key, cls in dico.items():
        if cls not in classes.values() and check(cls):
            classes[key] = cls
    return classes


def numericalOrder(nSteps, err):
    """
    Utility function to compute numerical order from error and nSteps vectors

    Parameters
    ----------
    nSteps : np.1darray or list
        Different number of steps to compute the error.
    err : np.1darray
        Diffenrent error values associated to the number of steps.

    Returns
    -------
    beta : float
        Order coefficient computed through linear regression.
    rmse : float
        The root mean square error of the linear regression.
    """
    nSteps = np.asarray(nSteps)
    x, y = np.log10(1/nSteps), np.log10(err)

    # Compute regression coefficients and rmse
    xMean = x.mean()
    yMean = y.mean()
    sX = ((x-xMean)**2).sum()
    sXY = ((x-xMean)*(y-yMean)).sum()

    beta = sXY/sX
    alpha = yMean - beta*xMean

    yHat = alpha + beta*x
    rmse = ((y-yHat)**2).sum()**0.5
    rmse /= x.size**0.5

    return beta, rmse
