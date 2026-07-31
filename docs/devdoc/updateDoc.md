# Update this documentation

📜 _If you think it can be clearer, or you want to add more details or tutorials ..._


## Generating local docs

First you need a few dependencies (besides those for `qmat`). For that download
the [source code](https://github.com/Parallel-in-Time/qmat) and install the package with all the
`docs` dependencies locally :

```bash
git clone https://github.com/Parallel-in-Time/qmat.git
cd qmat
pip install -e .[docs]  # on MAC-OS : pip install -e ".[docs]"
```

> 📜 The `-e` option ensures that your installed python package is directly linked to the sources (no copy of code),
> hence modifying any part of the source code (in particular the documentation)
> will be taken into account when `sphinx` will parse the code docstring.

Then to generate the documentation website locally, simply run :

```bash
cd docs
make html
```

This builds the `sphinx` documentation automatically in a `_build` folder,
and you can view it by opening `docs/_build/html/index.html` using your favorite browser.


## Updating a tutorial

When changing a [notebook tutorial](../notebooks), you should also regenerate it entirely, in particular if you modified parts of the code.
You can do that by running :

```bash
cd notebooks
./run.sh $NOTEBOOK_FILE
```

If you modified several notebooks, and as a safety, it is also possible to regenerate all doing :

```bash
./run.sh --all
```

> 📣 When modifying only the markdown text in a notebooks, it is not necessary to regenerate it.


## Adding a tutorial

In case a new major feature is added to qmat, it could be beneficial to add a dedicated tutorial for it.
To do that :

1. add the notebook in `docs/notebooks` folder following the prefix indexing of the previous notebook
2. add a link to it in `docs/notebooks.md` with a short description for it