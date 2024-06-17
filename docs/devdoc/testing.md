# Testing your changes

📜 _After doing some changes / corrections / addition in the code, you can run all the CI tests locally before any commit or PR._

## Install test dependencies

For reproducibility, it is recommended to use a dedicated environment to install all dependencies.
You can do that by running from `qmat` root folder :

```bash
$ python -m venv env
```

$\Rightarrow$ this will create a `env` folder in `qmat` root folder (ignored by `git`),
that you can activate using :

```bash
$ source ./env/bin/activate
```

> 🔔 In case you have the `base` `conda` environment as default on your computer, 
> you should deactivate it before activating `env` by running `conda deactivate`.  

If not already done, install all the test dependencies listed in the [pyproject.toml](../../pyproject.toml) file
under the `project.optional-dependencies` section.
Those can be installed one by one (if not already on your system),
or use this (dirty) shortcut by running from the `qmat` root folder :

```bash
$ pip install .[test]     # install qmat locally and all test dependencies
$ pip uninstall qmat      # remove the frozen qmat package installed locally
```

> 📣 Remember that the [recommended installation approach for developer](../installation) is to use a simple modification of the `PYTHONPATH` environment variable.

## Test local changes

The first thing to do (from the root `qmat` repo) is to run : 

```bash
$ python -c "import qmat"
```

This will trigger the [registration mechanism](./structure) that test the code structure at import, 
and ensures that all generators are correctly implemented 
(in particular, overriding of the correct methods, etc ...).

Then run the full test series with :

```bash
$ pytest -v ./tests
```

This will check :

- the basic generation of all registered $Q$-coefficients and $Q_\Delta$ approximations (using functions or generator objects)
- convergence order of all registered $Q$-coefficients
- some properties of all registered $Q_\Delta$ approximations

💡 **Hint :**

There is actually more than 3000 tests to check the package, that take around 1 minutes on a standard computer. 
So you may not want to run all of those every time you do a small modification somewhere 😅 ...
Here are a few tricks you can use :

```bash
$ pytest -v -x ./tests    # interrupt test on the first encountered error
$ pytest -v ./tests/test_1_nodes.py       # run only one test file
$ pytest -v ./tests/test_qcoeff           # run only one folder
$ pytest -v ./tests/test_1_nodes.py::testGauss            # run only one test function
$ pytest -v ./tests/test_1_nodes.py::testGauss[LEGENDRE]  # run only one test function with one given configuration
```

## Check code coverage

Once all test pass, you may check locally coverage by running (from the root folder) :

```bash
$ ./test.sh
$ coverage combine
$ python -m coverage html
```

This generates a html coverage report in `htmlcov/index.html` that you can read using your favorite web browser.

## Testing notebook tutorials

All notebooks are located in the [notebook docs folder](../notebooks). You can first check if they can be executed properly by running :

```bash
$ cd docs/notebooks
$ ./run-sh --all
```

💡 To execute only one notebook, simply run _e.g_ :

```bash
./run.sh 01_qCoeffs.ipynb
```

Finally, you can test all notebooks by running :

```bash
pytest ./ --nb-test-files -v
```

This will re-run each instructions in the notebooks, and compare if the generated outputs are identical to those of the locally stored notebook.