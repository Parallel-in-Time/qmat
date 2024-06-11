
# Installation

For now, you can only install the package locally by downloading the sources :

```bash
git clone https://github.com/Parallel-in-Time/qmat.git
```

And then, either add the code folder to your `PYTHONPATH` manually, _e.g_ :

```bash
cd qmat     # go into the local git repo
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

> 🔔 This is the recommended approach for developers, as any modification of your local `qmat` will be automatically taken into account

If you **only want to use the package**, you can simply use the `pip` local installer directly :

```bash
cd qmat     # go into the local git repo (if not already there ...)
pip install .
```

> 🛠️ Upload to `pypi` and `conda-forge` is still in construction ...
