# Add a Runge-Kutta scheme

Current $Q$-generators based on Runge-Kutta schemes are implemented in
{py:mod}`qmat.qcoeff.butcher`.
Those are based on Butcher tables from classical schemes available in the literature,
and the selected approach is to define **one class for one scheme**.

## Standard scheme

To add a new RK method, search first for its section in the `butcher.py` file, depending on its type
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

Here the `registerRK` decorator interfaces the classical `register` decorator for `QGenerator` classes,
but also :

1. checks if the dimensions of `A`, `b` and `c` are consistent
2. registers the generator in a specific category with all RK-type generators

> 💡 You can use either the built-in `list` or Numpy `nd.array` to add the class attributes `A`, `b` and `c`.

**Tip** : for large Butcher table, you can also use this approach (from the {py:class}`CashKarp <qmat.qcoeff.butcher.CashKarp>` class) :

```python
A = np.zeros((6, 6))
A[1, 0] = 1.0 / 5.0
A[2, :2] = [3.0 / 40.0, 9.0 / 40.0]
A[3, :3] = [0.3, -0.9, 1.2]
A[4, :4] = [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0]
A[5, :5] = [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0]
```

## Linear order

By default, a RK scheme is expected to show the same convergence order for linear autonomous
problems (_e.g_ the Dahlquist problem) as for non-linear ones, given by the `order` property.
But some schemes show super-convergence on linear problems, and in that case you should
override the `orderLinear` property from the base class :

```python
@registerRK
class NewRK(RK):
    # ...
    @property
    def orderLinear(self):
        return ...  # effective order for linear autonomous problems
```

> ⚠️ The `orderLinear` property must only be overridden when the scheme shows super-convergence
> on linear problems, _i.e_ when `orderLinear > order`.
> The value returned by `orderLinear` is checked by the convergence tests on the Dahlquist problem,
> while the one from `order` is checked on the (non-linear) Riccati problem.

## Convergence testing

To test your scheme ... you don't have to do anything 🥳 : all RK schemes are automatically tested
thanks to the [registration mechanism](./structure.md), that checks (in particular) the convergence
order of each scheme (global truncation error).

In particular, the order conditions of each Butcher table (for both the `order` and `orderLinear`
properties) are checked using the [physverify](https://github.com/rcgiri-physics/physverify) package,
developed by [Ram Chandra Giri](https://github.com/rcgiri-physics) 🙏

> ⚠️ Depending on the implemented RK scheme, convergence test may fail ... in that case no worries 😉 you'll just have to adapt your scheme to the test, as explained below ...

All convergence tests are done on the two following problems, using the order properties
of the scheme to set the expected convergence order (see the [linear order section](#linear-order)) :

- the (linear) Dahlquist problem, using `orderLinear` (or `orderLinearEmbedded` for the embedded method) :

```python
u0 = 1          # unitary initial solution
lam = 1j        # purely imaginary lambda
tEnd = 2*np.pi  # one time period
```

- the (non-linear) Riccati problem, using `order` (or `orderEmbedded` for the embedded method) :

```python
u0 = 1      # unitary initial solution
tEnd = 1
```

They use three numbers of time-steps for the convergence analysis, depending on the order of the method
(see [here ...](https://github.com/Parallel-in-Time/qmat/blob/main/tests/test_qcoeff/test_convergence.py#L10)).

But this automatic time-step size selection may not be adapted for methods with high error constant that require finer time-steps
to actually see the theoretical order.
In that case, simply add a `CONV_TEST_NSTEPS` _class attribute_ storing a list with **higher numbers of time-steps** in increasing order, high enough so the convergence test passes.

> 📜 See [SDIRK2_2 implementation](https://github.com/Parallel-in-Time/qmat/blob/e17e2dd2aebff1b09188f4314a82338355a55582/qmat/qcoeff/butcher.py#L269) for an usage example of `CONV_TEST_NSTEPS` ...


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
If this is not the case, then you should override the `orderEmbedded` property from the base class :

```python
@registerRK
class NewRK(RK):
    # ...
    @property
    def orderEmbedded(self):
        return ...  # effective embedded order
```

Similarly, the order of the embedded method for linear autonomous problems is defined
per default as **one order less than the linear order of the method** (that is, returned
by the `orderLinear` property). If the embedded method shows super-convergence on linear
problems, then override the `orderLinearEmbedded` property :

```python
@registerRK
class NewRK(RK):
    # ...
    @property
    def orderLinearEmbedded(self):
        return ...  # effective embedded order for linear autonomous problems
```

> ⚠️ Only override `orderLinearEmbedded` when the embedded method shows super-convergence
> on linear problems, _i.e_ when `orderLinearEmbedded > orderEmbedded`.
