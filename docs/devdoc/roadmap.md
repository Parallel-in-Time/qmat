# Development Roadmap

📜 _Planned steps for the package development ..._

Detailed description of all specific versions and their associated changes is available on the [Github Releases page](https://github.com/Parallel-in-Time/qmat/releases). 

**Status 3 - Alpha** : `v0.0.*`

- ✅ main structure and design
- ✅ full testing
- ✅ base documentation (notebook)
- ✅ documentation website (readthedocs)
- ✅ developer documentation
- ✅ pypi packaging and full installation doc
- ✅ github community compliance
- ✅ structure for embedded $Q$-coefficients

**Status 4 - Beta** : `v0.1.*`

- ✅ integration of all RK schemes from [pySDC](https://github.com/Parallel-in-Time/pySDC)
- ✅ citation file and zenodo releases
- integration of `qmat` into [pySDC](https://github.com/Parallel-in-Time/pySDC)
- full documentation of classes and functions
- finalization of extended usage tutorials ($S$-matrix, `dTau` coefficient for initial sweep, prolongation)
- ✅ full definition and documentation of the version update pipeline

**Status 5 - Production/Stable** : `v1.0.*`

- base console script interfacing `qmat` API for $Q$-coefficients and SDC coefficients generation (with IMEX)
- integration of `qmat` into [SWEET](https://gitlab.inria.fr/sweet/sweet)
- use of `qmat` for [Dedalus](https://github.com/DedalusProject/dedalus) IMEX SDC time-steppers developed within [pySDC](https://github.com/Parallel-in-Time/pySDC)
- distribution to other people using former version of the core `qmat` code (_e.g_ Alex Brown from Exeter, ...)
- addition of a few advanced usage tutorials :
    - `qmat` for non-linear ODE
    - multilevel SDC
    - PFASST

**Status 6 - Mature** : `v1.*.*`

- integration of SDC-Butcher theory from J. Fregin (with associated console scripts) 