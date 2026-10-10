"""Record the constructor arguments each functor instance was built with.

Functor classes come from C++ (pybind11) and do not store their constructor
arguments, so a compiled ``RollingMean`` cannot report its ``window_size``. This
module wraps every functor class in a thin, same-named Python subclass whose
``__init__`` calls the C++ constructor (which validates) and then records the
passed arguments, bound to their parameter names, on the instance as
``_screamer_params``. Everything else -- the per-sample compute, ``isinstance``
against the C++ class, and the Pipeline graph-building hook -- is inherited unchanged.

The captured params power three things: a readable ``repr``
(``RollingMean(window_size=20)``), Pipeline node labels, and Pipeline serialization.
"""
import json
import math
import re
import textwrap
import warnings
from importlib import resources

from . import screamer_bindings as _b

_HELP = None


def _load_help():
    """Load the parameter schema registry (screamer/data/help.json), once.

    A missing registry degrades gracefully (params captured positionally). A
    present-but-broken registry is a real problem, so it warns rather than
    silently losing every parameter name.
    """
    global _HELP
    if _HELP is None:
        try:
            with resources.files("screamer").joinpath("data/help.json").open() as fh:
                _HELP = json.load(fh)
        except FileNotFoundError:
            _HELP = {}
        except Exception as exc:  # corrupt JSON, permissions, ...
            warnings.warn(f"screamer: could not read help.json, functor parameter "
                          f"names will be unavailable: {exc}")
            _HELP = {}
    return _HELP


def _param_names(cls_name):
    """Ordered parameter names for a functor, or None if the schema is unknown."""
    entry = _load_help().get(cls_name)
    if not entry:
        return None
    return [p["name"] for p in entry.get("parameters", [])]


_DOCS_URL = "https://screamer.readthedocs.io/en/latest/"
_MD_HEADING_RE = re.compile(r"^(#{2,4}) (.+)$", re.M)
_MD_MARKER_RE = re.compile(r"^<!-- [A-Z_]+ -->\n?", re.M)


def _format_default(value):
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def build_docstring(cls_name):
    """A numpydoc-style docstring for ``cls_name`` from its help.json entry.

    The function pages under ``docs/functions_*`` are the single source: their
    frontmatter gives the summary and parameters, their body the details. None
    when the registry has no entry for ``cls_name``.
    """
    entry = _load_help().get(cls_name)
    if not entry:
        return None
    params = entry.get("parameters") or []
    signature = ", ".join(
        f"{p['name']}={_format_default(p['default'])}" if "default" in p else p["name"]
        for p in params)
    parts = [f"{cls_name}({signature})", "", entry.get("short", "").strip()]
    if params:
        parts += ["", "Parameters", "----------"]
        for p in params:
            head = p["name"]
            if p.get("type"):
                head += f" : {p['type']}"
            if "default" in p:
                head += f", default {_format_default(p['default'])}"
            parts.append(head)
            desc = (p.get("description") or "").strip()
            if p.get("enum"):
                desc += f" One of {', '.join(repr(v) for v in p['enum'])}."
            if desc:
                parts.append(textwrap.indent(textwrap.fill(desc.strip(), 76), "    "))
    details = _MD_MARKER_RE.sub("", entry.get("details") or "").strip()
    details = re.sub(r"\A## Description\s*\n", "", details)
    details = re.sub(r"\n{3,}", "\n\n", details)
    details = _MD_HEADING_RE.sub(lambda m: f"{m.group(2)}\n{'-' * len(m.group(2))}", details)
    if details:
        parts += ["", details]
    see_also = entry.get("see_also") or []
    if see_also:
        parts += ["", "See Also", "--------", ", ".join(see_also)]
    if entry.get("doc_path"):
        parts += ["", f"Full reference: {_DOCS_URL}{entry['doc_path']}.html"]
    return "\n".join(parts) + "\n"


def bind_params(cls_name, args, kwargs):
    """Bind positional ``args`` to parameter names and merge ``kwargs``.

    Uses the help.json schema for ``cls_name`` to name positional arguments. A
    functor absent from the schema keeps its positional arguments under an
    ``"args"`` key so nothing is lost.
    """
    names = _param_names(cls_name)
    params = {}
    if names is None:
        if args:
            params["args"] = list(args)
        params.update(kwargs)
        return params
    for i, value in enumerate(args):
        if i < len(names):
            params[names[i]] = value
        else:
            params.setdefault("args", []).append(value)
    params.update(kwargs)
    return params


def format_call(cls_name, params):
    """Render ``cls_name`` + params as a constructor call string."""
    parts = []
    for key, value in params.items():
        if key == "args":
            parts.extend(repr(v) for v in value)
        else:
            parts.append(f"{key}={value!r}")
    return f"{cls_name}({', '.join(parts)})"


def _make_wrapper(name, base):
    """Build a same-named subclass of ``base`` that captures its init args."""
    cls = type(name, (base,), {"__doc__": build_docstring(name), "__module__": "screamer"})

    def __init__(self, *args, **kwargs):
        super(cls, self).__init__(*args, **kwargs)
        self._screamer_params = bind_params(name, args, kwargs)

    def __repr__(self):
        return format_call(name, getattr(self, "_screamer_params", {}))

    # Keep the pybind constructor signature visible to introspection tools that
    # read __init__.__doc__ (e.g. devtools.get_constructor_arguments).
    __init__.__doc__ = getattr(base.__init__, "__doc__", None)
    cls.__init__ = __init__
    cls.__repr__ = __repr__
    cls._screamer_wrapped = True
    return cls


def _is_functor(obj):
    """True for a concrete functor class.

    Every functor (1-in/1-out ``ScreamerBase`` and multi-in ``FunctorBase``) derives
    from ``EvalOp``; the two abstract bases are excluded.
    """
    if not isinstance(obj, type):
        return False
    try:
        if not issubclass(obj, _b.EvalOp):
            return False
    except TypeError:
        return False
    return obj not in (_b.ScreamerBase, _b.EvalOp)


def install_param_capture(namespace):
    """Replace every functor class in ``namespace`` with a param-capturing subclass.

    Idempotent: already-wrapped classes are left alone.
    """
    for name, obj in list(namespace.items()):
        if not _is_functor(obj) or getattr(obj, "_screamer_wrapped", False):
            continue
        namespace[name] = _make_wrapper(name, obj)
