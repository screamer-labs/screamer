---
exempt_shape: true
---

# Install screamer

Install screamer from PyPI with pip. It needs Python 3.12 or newer, and one
wheel per platform covers 3.12 and every later 3.x.

```console
pip install screamer
```

The wheel includes the compiled C++ extension, and its one runtime dependency
is numpy.

## Packages for the examples

The examples in this Guide also use pandas and matplotlib:

```console
pip install pandas matplotlib
```

## From source

Clone the repository and install it in editable mode:

```console
git clone https://github.com/screamer-labs/screamer.git
cd screamer
pip install -e ".[test]"
make test
```

The `test` extra adds pytest and the reference libraries the tests compare
against, such as SciPy and TA-Lib.
`make install-dev` rebuilds the C++ extension and installs it in editable mode,
and `make test` runs that step and then the test suite.

## Go deeper

- [Apply a function to an array, a column and a live stream](first-function.md): the first example.
- [Performance](../../performance.md): how fast the functions run.
