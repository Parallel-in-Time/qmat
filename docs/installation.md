
# Installation

## Using PyPI

You can download the latest version from [`pypi`](https://pypi.org/) :

```bash
pip install qmat
```

## Using conda

Currently, no version is distributed on conda-forge. However using `pip` from `conda` will install `qmat` in your conda environment.

If you are using a `environment.yml` file with conda, then you can add it as a dependency like this :

```yaml
name: yourEnv
channels:
  - conda-forge
  - defaults
dependencies:
  ...
  - pip
  - pip:
    qmat
```

## Install from source

In case you want the latest revision (or a specific branch), you can directly clone the sources from `github` :

```bash
$ git clone https://github.com/Parallel-in-Time/qmat.git
```

If you **want to use the package only**, simply use the `pip` local installer directly :

```bash
$ cd qmat     # go into the local git repo
$ pip install .
```

For **developers who want to contribute**, recommended approach is to add 
the code folder to your `PYTHONPATH` (if not done already by your IDE), _e.g_ :

```bash
$ cd qmat     # go into the local git repo (if not already there)
$ export PYTHONPATH=$PYTHONPATH:$(pwd)
```

> 🔔 Using `$ pip install -e .` is also possible for developments, but then you have a persistent installation that you should be aware of ...





