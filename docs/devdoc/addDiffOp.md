# Add a differential operator

📜 _Solvers implemented in {py:mod}`qmat.solvers.generic` can be used_
_with others {py:class}`DiffOp <qmat.solvers.generic.DiffOp>` classes_
_than those implemented in {py:mod}`qmat.solvers.generic.diffops`._

To add a new one, implement it at the end of the `diffops.py` module,
using the following template :

```python

@registerDiffOp
class Yoodlidoo(DiffOp):
    r"""
    Base description, in particular its equation :

    .. math::

        \frac{du}{dt} = ...

    And some parameters description ...
    """
    def __init__(self, params="value"):
        self.params = params
        u0 = np.array([1, 0], dtype=float)
        super().__init__(u0)

    def evalF(self, u, t, out):
        # TODO : your implementation
        pass
```

And that's all ! The `registerDiffOp` operator will automatically
- add your class in the `DIFFOPS` dictionary to make it generically available
- check if your class override properly the `evalF` function (import error if not)
- add your class to the [CI tests](./testing.md)

> 📣 Per default, all `DiffOp` classes must be instantiable with default parameters
> in order to run the tests (see the {py:func}`DiffOp.test <qmat.solvers.generic.DiffOp.test>`
> class method). But you can change that by overriding the `test` class method and put your own
> preset parameters for the test (checkout the
> {py:func}`ProtheroRobinson <qmat.solvers.generic.diffops.ProtheroRobinson>` class for an example).

Finally, the `DiffOp` class implements a default `fSolve` method,
but you can also implement a more efficient approach tailored to your problem like this :

```python
@registerDiffOp
class Yoodlidoo(DiffOp):
    # ...

    def fSolve(self, a:float, rhs:np.ndarray, t:float, out:np.ndarray):
        # TODO : your ultra-efficient implementation that will be
        #        way better than a generic call of scipy.optimize.fsolve
        #        or scipy.optimize.newton_krylov.
        pass
```