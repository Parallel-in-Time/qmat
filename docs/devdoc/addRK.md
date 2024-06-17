# Add a Runge-Kutta scheme

Current $Q$-generators based on Runge-Kutta schemes are implemented in the 
[`qmat.qcoeff.butcher`](https://github.com/Parallel-in-Time/qmat/blob/main/qmat/qcoeff/butcher.py) submodule.
Those are based on Butcher tables from classical schemes available in the literature, 
and the selected approach is to define **one class for one scheme**.

## Standard scheme

In order to add a new RK, search first for its section in the `butcher.py` file, depending on its type 
(explicit or implicit) and its order. Then add a new class at the bottom of this section following this template :

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

> 💡 You can use either the built-in `list` or Numpy `nd.array` to add the class attributes `A`, `b` and `c`.
> For large Butcher table, you can also use this approach (from the `CashKarp` class) :
>
> ```python
> A = np.zeros((6, 6))
> A[1, 0] = 1.0 / 5.0
> A[2, :2] = [3.0 / 40.0, 9.0 / 40.0]
> A[3, :3] = [0.3, -0.9, 1.2]
> A[4, :4] = [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0]
> A[5, :5] = [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0]
> ```

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

## Embedded scheme

Butcher tables with additional coefficients for embedded method can also be added to new or existing RK schemes.
For that, simply define a `b2` class attribute :

```python
@registerRK
class NewRK(RK):
    """Some new RK method from ..."""
    ## previous coefficients ... 
    b2 = ... # embedded coefficients
```

Per default, $Q$-generators define the order of the embedded method (using those additional coefficient)
as **one order less than the method's order** (that is, returned by the `order` property).
If this is not the case, then you should override the `weightEmbedded` property from the base class :

```python
@registerRK
class NewRK(RK):
    @property
    def weightsEmbedded(self):
        return ...  # effective embedded order
```

