"""Build the Cython event loops used by the streaming benchmark.

    python benchmarks/build_compiled.py build_ext --inplace

The streaming suite runs without this: the compiled variants are skipped when
the extension is not importable, the same way the batch suite skips TA-Lib when
it is not installed.
"""
import numpy as np
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:                                  # pragma: no cover
    raise SystemExit("Cython is needed to build the compiled benchmark loops")

setup(
    name="streaming_compiled",
    ext_modules=cythonize(
        [Extension("streaming_compiled", ["streaming_compiled.pyx"],
                   include_dirs=[np.get_include()])],
        language_level=3,
    ),
)
