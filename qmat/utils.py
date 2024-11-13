#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package utility module
"""
import inspect
import pkgutil

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


def checkGenericConstr(cls):
    sig = inspect.signature(cls.__init__)
    try:
        par = sig.parameters["kwargs"]
        assert par.kind == par.VAR_KEYWORD
    except (KeyError, AssertionError):
        raise AssertionError(f"{cls.__name__} class requires **kwargs in its constructor")


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
