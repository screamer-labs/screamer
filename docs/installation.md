# Installation

This page covers the prerequisites and installation options for screamer.

## Prerequisites

Python 3.11 or newer is required (`requires-python = ">=3.11"`; Python 3.11
through 3.14 are supported).

## Installing screamer

```console
pip install screamer
```

The wheel includes the compiled C++ extension. The only runtime dependency is
`pybind11>=2.9`, which is bundled in the wheel and does not need a separate
install step.

## Running the example notebooks

The example notebooks additionally require numpy, and some also use matplotlib
and pandas. Install them all at once with the `docs` extra:

```console
pip install "screamer[docs]"
```

Or install only what you need directly:

```console
pip install numpy matplotlib pandas
```

## From source / development

Clone the repository and do an editable install:

```console
git clone https://github.com/screamer-labs/screamer.git
cd screamer
pip install -e .
```

To run the test suite, add the `test` extra (pytest, numpy>=2.0, scipy,
pandas, PyYAML):

```console
pip install -e ".[test]"
make test
```

The project ships `make` targets for the full build cycle. `make install-dev`
rebuilds the C++ extension and installs it in editable mode; `make test` runs
that step then runs the suite.
