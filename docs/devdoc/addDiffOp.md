# Add a differential operator

📜 _Solvers implemented in {py:mod}`qmat.solvers.generic` can be used_
_with other {py:class}`DiffOp <qmat.solvers.generic.DiffOp>` classes_
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
        # use some initialization parameters
        u0 = ... # define your initial vector
        super().__init__(u0)

    def evalF(self, u, t, out:np.ndarray):
        r"""
        Evaluate :math:`f(u,t)` and store the result into `out`.

        Parameters
        ----------
        u : np.ndarray
            Input solution for the evaluation.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Output array in which is stored the evaluation.
        """
        out[:] = ... # put the result into out
```

And that's all ! The `registerDiffOp` operator will automatically
- add your class in the `DIFFOPS` dictionary to make it generically available
- check if your class properly overrides the `evalF` function (import error if not)
- add your class to the [CI tests](./testing.md)

> 📣 Per default, all `DiffOp` classes must be instantiable with default parameters
> in order to run the tests (see the {py:func}`DiffOp.test <qmat.solvers.generic.DiffOp.test>`
> class method). But you can change that by overriding the `test` class method and put your own
> preset parameters for the test (checkout the
> {py:func}`ProtheroRobinson <qmat.solvers.generic.diffops.ProtheroRobinson>` class for an example).

Finally, the `DiffOp` class implements a default `fSolve` method, that solves :

$$
u - \alpha f(u, t) = rhs
$$

for any given $\alpha, t, rhs$.
It relies on generic non-linear root-finding solvers, namely `scipy.optimize.fsolve` for small problems 
and `scipy.optimize.newton_krylov` for large scale problems.
You can also implement a more efficient approach tailored to your problem like this :

```python
@registerDiffOp
class Yoodlidoo(DiffOp):
    # ...

    def fSolve(self, a:float, rhs:np.ndarray, t:float, out:np.ndarray):
        r"""
        Solve :math:`u-\alpha f(u,t)=rhs` for given :math:`u,t,rhs`,
        using `out` as initial guess and storing the final result into it.

        Parameters
        ----------
        a : float
            The :math:`\alpha` coefficient.
        rhs : np.ndarray
            The right hand side.
        t : float
            Time for the evaluation.
        out : np.ndarray
            Input-output array used as initial guess,
            in which is stored the solution.
        """
        # TODO : your ultra-efficient implementation that will be
        #        way better than a generic call of scipy.optimize.fsolve
        #        or scipy.optimize.newton_krylov.
        out[:] = ...
```

> 🔔 Note that `out` will be used as output for the solution, 
> but its input value can also be used as initial guess for any
> iterative solver you may want to use.