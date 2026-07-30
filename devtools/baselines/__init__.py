"""Reference implementations, loaded by file name.

Every module in this directory is imported and its classes are re-exported, so
`devtools.baselines.RollingMean_pandas` resolves without an explicit import.

A baseline may depend on a library screamer itself does not require. TA-Lib is
the case that matters: it needs a C library installed first, so `pyproject.toml`
keeps it in an opt-in extras group and CI does not install it. A baseline whose
dependency is missing is skipped and recorded in `unavailable`, rather than
raising out of this loop.

That distinction is deliberate. Importing this package is what `tests/param_cases.py`
does at collection time, so an unhandled ImportError here takes down the whole
test suite in any environment lacking one optional library. A skipped baseline
instead means that operator has no reference in this environment, which
`get_baselines` reports as "none" and the coverage gate reports as a gap.

Only ImportError is swallowed. A baseline that raises anything else is broken
and must fail loudly.
"""
import importlib
import os

# module name -> the ImportError message, for baselines that could not load.
unavailable: dict[str, str] = {}

for filename in sorted(os.listdir(os.path.dirname(__file__))):
    if not filename.endswith(".py") or filename == "__init__.py":
        continue
    module_name = filename[:-3]
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
    except ImportError as exc:
        unavailable[module_name] = str(exc)
        continue
    globals().update({
        name: getattr(module, name)
        for name in dir(module)
        if isinstance(getattr(module, name), type)
    })


def missing_dependencies() -> dict[str, str]:
    """Baselines skipped in this environment, mapped to why."""
    return dict(unavailable)
