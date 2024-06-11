# General code structure

📜 _Quick introduction on how the code is designed and organized ..._

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

## $Q$-generators implementation

To implement a new $Q$-generator (in an existing or new category) new classes must at least follow this template :

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
```

The `nodes`, `weights`, and `Q` properties have to be overridden 
(`register` actually raises an error if not) and return 
the expected arrays in `numpy.ndarray` format :

1. `nodes` : 1D vector of size `nNodes`
2. `weights` : 1D vector of size `nNides`
3. `Q` : 2D matrix of size `(nNodes,nNodes)`

While `nNodes` is a variable depending on the generator instance, later on the tests checks if the for each $Q$-generators, dimensions of `nodes`, `weights` and `Q` are consistent.

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





🛠️ ... in construction

## $Q_\Delta$-generators implementation

🛠️ ... in construction

## Additional submodules

🛠️ ... in construction