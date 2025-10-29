# Add a $\phi$-based time-integrator

📜 _Additional time schemes can be added using the [$\phi$ formulation](../notebooks/14_phiIntegrator.ipynb)_
_to test other variants of $Q_\Delta$-coefficients free Spectral Deferred Correction._
_For that, you can implement a new {py:mod}`PhiSolver <qmat.solvers.generic.PhiSolver>` class in the {py:mod}`qmat.solvers.generic.integrators` module_.

Add your class at the end of the `qmat.solvers.generic.integrators.py` module using the following template :

```python
class Phidlidoo(PhiSolver):
    r"""
    Base description, in particular its definition :

    .. math::

        \phi(u_0, u_1, ..., u_{m}, u_{m+1}) =
            ...

    And eventual parameters description ...
    """

    def evalPhi(self, uVals, fEvals, out, t0=0):
        m = len(uVals) - 1
        assert m > 0
        assert len(fEvals) in [m, m+1]

        # TODO : integrators implementation
```

The first lines are not mandatory, but ensure that the `evalPhi` is properly evaluated.

> 📣 New `PhiSolver` classes are not automatically tested, so you'll have to write
> some dedicated test for your new class in `tests.test_solvers.test_integrators.py`.
> Checkout those already implemented for `ForwardEuler` and `BackwardEuler`.

As for the {py:class}`DiffOp <qmat.solvers.generic.DiffOp>` class,
the {py:class}`PhiSolver <qmat.solvers.generic.PhiSolver>` implement a generic default
`phiSolve` method, that you can override by a more efficient specialized approach.

> 💡 Note that the model above inherits the `__init__` constructor of the `PhiSolver` class,
> so it can take any `DiffOp` class as parameter.
> If your time-integrator is specialized for some kind of differential operators
> (_e.g_ a semi-Lagrangian scheme for an advective problem),
> then you probably need to override the `__init__` method in your class too.
