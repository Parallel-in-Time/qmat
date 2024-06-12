
# Installation

For now, you can only install the package locally by downloading the sources.

```bash
$ git clone https://github.com/Parallel-in-Time/qmat.git
```

If you **want to use the package only**, simply use the `pip` local installer directly :

```bash
$ cd qmat     # go into the local git repo
$ pip install .
```

> 🛠️ Upload to `pypi` and `conda-forge` is still in construction ...

For **developers who want to contribute**, recommended approach is to add 
the code folder to your `PYTHONPATH` (if not done already by your IDE), _e.g_ :

```bash
$ cd qmat     # go into the local git repo (if not already there)
$ export PYTHONPATH=$PYTHONPATH:$(pwd)
```

> 🔔 Using `$ pip install -e .` is also possible for developments, but then you have a persistent installation that you should be aware of ...





