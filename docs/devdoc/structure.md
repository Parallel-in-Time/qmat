# General code structure

📜 _Quick introduction on the code design and how to extend it ..._

## Generator registration

The two main features, namely the generation of $Q$-coefficients and $Q_\Delta$ approximations,
are respectively implemented in the `qmat.qcoeff` and `qmat.qdelta` sub-packages.
Different categories of generators are implemented in dedicated submodules of their respective sub-packages,
_e.g_ : 

- `qmat.qcoeff.collocation` for Collocation-based $Q$-generators 
- `qmat.qdelta.algebraic` for algebraic based $Q_\Delta$ approximations
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
    def nodes(self):
        # TODO : implementation

    @property
    def weights(self):
        # TODO : implementation

    @property
    def Q(self):
        # TODO : implementation

    @property
    def order(self):
        # TODO : implementation
```

The `nodes`, `weights`, and `Q` properties have to be overridden 
(`register` actually raises an error if not) and return 
the expected arrays in `numpy.ndarray` format :

1. `nodes` : 1D vector of size `nNodes`
2. `weights` : 1D vector of size `nNides`
3. `Q` : 2D matrix of size `(nNodes,nNodes)`

While `nNodes` is a variable depending on the generator instance, later on the tests checks if the for each $Q$-generators, dimensions of `nodes`, `weights` and `Q` are consistent.
Finally, you should implement the `order` property, that returns the theoretical accuracy order of the associated scheme (global truncation error).
The later is used in the [test series for convergence](https://github.com/Parallel-in-Time/qmat/blob/main/tests/test_qcoeff/test_convergence.py) to check the coefficients.

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

You can provide required parameters (like `param1`) or optional ones with default value (like `param2`).

> ⚠️ For required parameters, you must provide a default value in the class attribute `DEFAULT_PARAMS`, such that the `QGenerator.getInstance()` class method works.
> The later is used by to create a default instance of the $Q$-generator, by setting required parameters values using `DEFAULT_PARAMS`.

After implementing a new generator, you should test is by running the following test :

```
pytest -v ./tests/test_qcoeff
```

This will run all consistency and convergence check tests on all generators (including yours), more details on how to run the tests are provided [here ...](./testing.md)

## $Q_\Delta$-generators implementation

First, know that the base `QDeltaGenerator` class implement the following constructor :

```python
class QDeltaGenerator(object):

    def __init__(self, Q, **kwargs):
        self.Q = np.asarray(Q, dtype=float)
        self.QDelta = np.zeros_like(self.Q)
```
This default constructor is actually used by all the specialized generators
implemented in `qmat.qdelta.algebraic`, as their $Q_\Delta$ approximation is build directly from
the $Q$ matrix given as parameter.

To implement a new $Q_\Delta$-generator (in an existing or new category), new classes must at least follow this template :

```python
from qmat.qdelta import QDeltaGenerator, register

@register
class MyGenerator(QDeltaGenerator):

    def getQDelta(self, k=None):
        # TODO : implementation
        return self.QDelta
```

In practice, `getQDelta` must modify the current `QDelta` attribute (initialized with zeros) and return it. 
You may implement a check avoiding to recompute `QDelta` when already computed, _e.g_

```python
@register
class MyGenerator(QDeltaGenerator):

    def getQDelta(self, k=None):
        if hasattr(self, "_computed"):
            return self.QDelta
        # TODO : implementation
        self._computed = "ouiii"
        return self.QDelta
```

> ⚠️ Even if this may not be used by your generator, the `getQDelta` method should always take a `k` optional parameter (with the default value you see fit, `None` is enough if you don't use `k`).

You can also redefine the constructor of your generator like this :
```python
@register
class MyGenerator(QDeltaGenerator):

    def __init__(self, param1, param2, **kwargs):
        # TODO : implementation
        self.QDelta = ...
```

But then it is necessary to :

1. add the `**kwargs` arguments to your constructor, but don't use it for your generator's parameters : `**kwargs` is only used when series of $Q_\Delta$ matrices are generated from different types of generators
2. zero-initialize a `QDelta` `numpy.ndarray` with the appropriate shape (square matrix).


## Additional submodules

Several "utility" modules are available in `qmat` :

- [`qmat.nodes`](https://github.com/Parallel-in-Time/qmat/blob/main/qmat/nodes.py) : implement a `NodesGenerator` class for node generation with various distributions
- [`qmat.lagrange`](https://github.com/Parallel-in-Time/qmat/blob/main/qmat/lagrange.py) : implement a `LagrangeApproximation` class used to compute weights and $Q$ matrix for collocation, interpolation coefficients, ...
- [`qmat.sdc`](https://github.com/Parallel-in-Time/qmat/blob/main/qmat/sdc.py) : basic generic SDC solvers that can be used for first experiments and tests
- [`qmat.utils`](https://github.com/Parallel-in-Time/qmat/blob/main/qmat/utils.py) : as the name of the submodule suggest ...