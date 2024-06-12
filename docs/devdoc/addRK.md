# Add a Runge-Kutta scheme

Current $Q$-generators based on Runge-Kutta schemes are implemented in the `qmat.qcoeff.butcher` submodule.
Those are based on Butcher tables from classical schemes available in the literature, 
and the selected approach is to define **one class for one scheme**.

In order to add a new one, you can add a new class at the bottom of the module following this template :

```python
@registerRK
class NewRK(RK):
    """Some new RK method from ..."""
    A = ... # TODO
    b = ... # TODO
    c = ... # TODO

    @property
    def order(self): return ... # TODO
```

Here the `registerRK` decorators interfaces the classical `register` decorator for `QGenerator` classes,
but also :

1. check if the dimensions of the `A`, `b` and `c` are consistent
2. register the generator in a specific category with all RK-type generators

Finally, for testing it ... you don't have to do anything 🥳 : all RK schemes are automatically tested 
thanks to the [registration mechanism](./structure.md).

> ⚠️ All convergence tests are done on a given Dahlquist problem :
>
> ```python
> u0 = 1        # unitary initial solution
> lam = 1j      # purely imaginary lambda
> T = 2*np.pi   # one time period
> ```
> 
> using three numbers of time-steps for the convergence analysis, depending on the order of the method 
> (see [tests/test_coeff/test_convergence.py](https://github.com/Parallel-in-Time/qmat/blob/main/tests/test_qcoeff/test_convergence.py#L10)).
> But this automatic convergence testing may not be adapted for methods with high error constant that require finer time-steps
> to actually see the theoretical order. 

In case you are implementing a RK method with high error constant, you may need to take more time-steps than those selected automatically 
from the order. To do that, simply add the list of number of time-steps in a `CONV_TEST_NSTEPS` class attribute (in increasing order), 
see [SDIRK2_2 implementation](https://github.com/Parallel-in-Time/qmat/blob/e17e2dd2aebff1b09188f4314a82338355a55582/qmat/qcoeff/butcher.py#L259) for an example ...