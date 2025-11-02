# Code structure

📜 _Quick introduction on how the package is designed and how to extend it ..._

## Registration mechanism

The two main features, namely the generation of $Q$-coefficients and $Q_\Delta$ approximations,
are respectively implemented in the {py:mod}`qmat.qcoeff` and {py:mod}`qmat.qdelta` sub-packages.
Different categories of generators are implemented in dedicated submodules of their respective sub-packages,
_e.g_ :

- {py:mod}`qmat.qcoeff.collocation` for Collocation-based $Q$-generators
- {py:mod}`qmat.qdelta.algebraic` for algebraic based $Q_\Delta$ approximations
- ...

Each sub-package contains a `__init__.py` file implementing the generic parent class for all generators.
In their submodules, generators are implemented using a **registration mechanism**,
_e.g_ for the Collocation-based $Q$-generators :

```python
from qmat.qcoeff import QGenerator, register

@register
class Collocation(QGenerator):
    aliases = ["coll"]
    # ...
```

A similar mechanism is used for $Q_\Delta$ generators. The `register` function is used as class decorator which :

- checks that the implemented class properly overrides the method of its parent class (more specific details below)
- stores it in a centralized dictionary allowing a quick access using the class name or one of its aliases :
    - `qmat.Q_GENERATORS` for $Q$-coefficients
    - `qmat.QDELTA_GENERATORS` for the $Q_\Delta$ approximations

> 💡 Different aliases for the generator can be provided with the `aliases` class attribute, but are not mandatory (defining the class attribute is optional).

## $Q$-generators implementation

To implement a new $Q$-generator (in an existing or new category), new classes must at least follow this template :

```python
from qmat.qcoeff import QGenerator, register

@register
class MyGenerator(QGenerator):

    @property
    def nodes(self)
        # TODO : returns a np.1darray

    @property
    def weights(self):
        # TODO : returns a np.1darray

    @property
    def Q(self):
        # TODO : returns a np.2darray:

    @property
    def order(self):
        # TODO : returns an int
```

The `nodes`, `weights`, and `Q` properties have to be overridden
(`register` actually raises an error if not) and return
the expected arrays in `numpy.ndarray` format :

1. `nodes` : 1D vector of size `nNodes`
2. `weights` : 1D vector of size `nNodes`
3. `Q` : 2D matrix of size `(nNodes,nNodes)`

While `nNodes` is directly determined from the `nodes` property, later on the tests checks if the for each $Q$-generators, dimensions of `nodes`, `weights` and `Q` are consistent.
Finally, you should implement the `order` property, that returns the theoretical accuracy order of the associated scheme (global truncation error).
Value returned by `order` is used in the [test series for convergence](https://github.com/Parallel-in-Time/qmat/blob/main/tests/test_qcoeff/test_convergence.py) to check the coefficients.

> 🔔 For Runge-Kutta type generators, their implementation use an additional abstract layer to simplify the addition
> of new schemes, see [specific documentation to add RK schemes ...](./addRK.md)

Even if not it's not mandatory, $Q$-generators can implement a constructor to store parameters, _e.g_ :

```python
from qmat.qcoeff import QGenerator, register

@register
class MyGenerator(QGenerator):

    DEFAULT_PARAMS = {
        "param1": 0.5,
    }

    def __init__(self, param1, param2=1):
        self.param1 = param1
        self.param2 = param2

    # Implementation of nodes, weights and Q properties
```

You can provide required parameters (_e.g_ `param1`) or optional ones with default value (_e.g_ `param2`).

> ⚠️ For required parameters, you must provide a default value in the class attribute `DEFAULT_PARAMS`, such that the `QGenerator.getInstance()` class method works.
> The later is used during testing to create a default instance of the $Q$-generator, by setting required parameters values using `DEFAULT_PARAMS`.

After implementing a new generator, you should test is by running the following test :

```
pytest -v ./tests/test_qcoeff
```

This will run all consistency and convergence check tests on all generators (including yours), more details on how to run the tests are provided [here ...](./testing.md)

> 🔔 Convergence tests for new $Q$-generators are automatically done depending on its order. In some particular case, you may
> have to add a `CONV_TEST_NSTEPS` class variable to your generator class for those tests to pass
> (_e.g_, if your generator has a high error constant).
> See [documentation on adding RK schemes](./addRK.md#convergence-testing) for more details ...

## $Q_\Delta$-generators implementation

By default, the base `QDeltaGenerator` class implement those base methods, that may be used by any
specialized $Q_\Delta$ generator.

```python
class QDeltaGenerator(object):

    def __init__(self, Q, **kwargs):
        self.Q = np.asarray(Q, dtype=float)

    @property
    def size(self):
        return self.Q.shape[0]

    @property
    def zeros(self):
        M = self.size
        return np.zeros((M, M), dtype=float)
```

The default constructor stores the $Q$ matrix that is approximated,
and the `size` property is used to determine the shape of generated $Q_\Delta$ approximation,
and the `zeros` property can be used to generate the initial basis for $Q_\Delta$.

> 🔔 The default constructor is used by all the specialized generators implemented in `qmat.qdelta.algebraic`,
> as their $Q_\Delta$ approximation is build directly from the $Q$ matrix given as parameter.


To implement a new $Q_\Delta$-generator (in an existing or new category), new classes must at least follow this template :

```python
from qmat.qdelta import QDeltaGenerator, register

@register
class MyGenerator(QDeltaGenerator):

    def computeQDelta(self, k=None):
        # TODO : returns a np.2darray with shape (self.size, self.size)
```

The `computeQDelta` must simply returns the $Q_\Delta$ approximation for this generator,
potentially using the `zeros` property as starting basis.

**📣 Important :** even if this may not be used by your generator, the `computeQDelta` method **must always**
take a `k` optional parameter corresponding to a **sweep or iteration number** in SDC or iterated RK methods,
starting at $k=1$ for the first sweep.
The default value for this parameter must be :

- `None` if $Q_\Delta$ does not vary with `k`
- **any other value** you see fit if $Q_\Delta$ varies with `k`. For instance, using `1` as default value :

```python
def computeQDelta(self, k=1):
    if k is None: k=1
    # TODO : returns a np.2darray with shape (self.size, self.size)
```

> ⚠️ The `computeQDelta` method must be able to take `k=None` as argument, and potentially replace it by its default value.

You can also redefine the constructor of your generator like this :
```python
@register
class MyGenerator(QDeltaGenerator):

    def __init__(self, param1, param2, **kwargs):
        # TODO : implementation

    @property
    def size(self):
        # TODO : proper redefinition
```

But then it is necessary to :

1. add the `**kwargs` arguments to your constructor, but don't use it for your generator's parameters : `**kwargs` is only used when $Q_\Delta$ matrices are generated from different types of generators using one single call
2. properly redefine the `size` property **if you don't store any** $Q$ **matrix attribute** in your constructor

## Additional sub-packages

- {py:mod}`qmat.solvers` : implements various generic ODE making use of `qmat`-generated coefficients. Can be modified to [add new differential operators](./addDiffOp.md) or [add new $\phi$-based integrators](./addPhiIntegrator.md)
- {py:mod}`qmat.playgrounds` : can be modified to [add a playground](./addPlayground.md), _i.e_ non-tested experiments or examples script

## Additional submodules

- {py:mod}`qmat.nodes` : can be modified to add new functionalities to the `NodesGenerator` class, or improve the current implementations
- {py:mod}`qmat.lagrange` : can be modified to add new functionalities to the `LagrangeApproximation` class, or improve the current implementations
- {py:mod}`qmat.mathutils` : can be modified to add additional mathematical utility functions used by some parts in `qmat` (like array operations, regression tools, etc ...)
- {py:mod}`qmat.utils` : can be modified to add additional (non mathematical) utility functions used by some parts in `qmat` (like timers, implementation check functions, etc ...)