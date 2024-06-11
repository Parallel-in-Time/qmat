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

🛠️ ... in construction

## $Q_\Delta$-generators implementation

🛠️ ... in construction

## Additional submodules

🛠️ ... in construction