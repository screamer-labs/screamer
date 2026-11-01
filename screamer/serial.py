"""Save and load screamer operators.

An operator's file holds its settings (the constructor arguments) and its
state (what it has learned from the data so far), so a loaded operator
continues exactly where the saved one stopped::

    f = screamer.RollingMean(20)
    for x in history:
        f(x)
    f.save("mean.json")
    g = screamer.load("mean.json")   # continues where f stopped

Every operator has these methods:

* ``save(path)`` writes the file: CBOR for a ``.cbor`` path, JSON otherwise.
* ``dumps(format="json")`` returns the file as ``str``, or as ``bytes`` for
  ``format="cbor"``.
* ``settings()`` returns the constructor arguments as a dict.
* ``state()`` returns the state, ``{"version": N, "values": {...}}``, and
  ``set_state(d)`` loads one.

A default batch call starts from a reset operator and leaves it reset.
``resume(data)`` is the batch call that starts from the operator's state and
keeps the end state, so ``f.resume(history)`` warms ``f`` up for saving or for
calls with one event at a time. On a (T, K) array ``resume`` keeps one state
per column, saved as ``{"version": N, "columns": [{...}, ...]}``; a call with
one event then raises until ``reset()`` or a default batch call drops it.

``load``, ``loads`` and ``from_settings`` build operators. The C++ core writes
and reads the files (``include/screamer/serial``), so Python, JavaScript and
C++ write the same bytes for the same operator. An operator is built through
its screamer class, so a loaded one has the repr and ``Pipeline`` behavior of
one built by hand.

The errors are ``SaveError`` (a ``ValueError``) and its subclasses
``StateVersionError``, ``NotSaveable`` and ``StateMismatch``. Each message
starts with the path of the place it is about, such as
``RollingMean.buffer``.
"""
import json
import os
import warnings

from . import screamer_bindings as _b

SaveError = _b._SaveError
StateVersionError = _b._StateVersionError
NotSaveable = _b._NotSaveable
StateMismatch = _b._StateMismatch
for _cls in (SaveError, StateVersionError, NotSaveable, StateMismatch):
    _cls.__module__ = "screamer"
del _cls

__all__ = ["load", "loads", "from_settings", "SaveError", "StateVersionError",
           "NotSaveable", "StateMismatch"]

_ON_MISMATCH = ("error", "fresh")


def load(path, on_mismatch="error"):
    """Load an operator from a file written by ``save``.

    Parameters
    ----------
    path : str or os.PathLike
        The file, JSON or CBOR (the encoding is read from the file).
    on_mismatch : {"error", "fresh"}
        What to do with a state that does not fit the file's settings (an
        edited setting) or that another state version wrote: raise, or keep
        the settings, start the operator fresh and warn.
    """
    with open(os.fspath(path), "rb") as fh:
        return loads(fh.read(), on_mismatch=on_mismatch)


def loads(data, on_mismatch="error"):
    """Load an operator from the ``str`` or ``bytes`` that ``dumps`` returned.

    See ``load`` for ``on_mismatch``.
    """
    if on_mismatch not in _ON_MISMATCH:
        raise ValueError(f"on_mismatch must be 'error' or 'fresh', got {on_mismatch!r}")
    type_name, settings, state = _b._serial_load_node(data)
    op = _build(type_name, json.loads(settings), "object")
    if state is not None:
        try:
            op._set_state_json(state, type_name)
        except (StateMismatch, StateVersionError) as e:
            if on_mismatch == "error":
                raise
            # A failed load leaves the operator as built: fresh.
            warnings.warn(f"{e}; {type_name} starts fresh", UserWarning, stacklevel=2)
    return op


def from_settings(node):
    """A fresh operator from ``{"type": T, "settings": {...}}``.

    ``node`` is the ``"object"`` of a save file, or a dict built from an
    operator's ``settings()``. A ``"state"`` in it is ignored.
    """
    if not isinstance(node, dict):
        raise SaveError(f"expected {{'type': ..., 'settings': {{...}}}}, got {type(node).__name__}")
    return _build(node.get("type"), node.get("settings", {}), "object")


def _build(type_name, settings, path):
    """The operator ``type_name(**settings)``; ``path`` names it in errors
    ("object" for the top of a file, else the setting that holds it)."""
    import screamer
    from ._functor_params import _load_help
    cls = getattr(screamer, type_name, None) if isinstance(type_name, str) else None
    if not getattr(cls, "_screamer_wrapped", False):
        raise SaveError(f"{path}.type: unknown type {type_name!r}")
    label = type_name if path == "object" else path
    if not isinstance(settings, dict):
        raise SaveError(f"{label}.settings: expected an object, got {type(settings).__name__}")
    settings = _renamed(type_name, settings, label)
    params = {p["name"]: p for p in (_load_help().get(type_name) or {}).get("parameters", [])}
    kwargs = {}
    for name, value in settings.items():
        if name not in params:
            raise SaveError(f"{label}.settings.{name}: unknown parameter")
        if value is None:            # an optional argument that was not given
            continue
        if params[name].get("type") == "op":
            value = _build_op(value, f"{label}.settings.{name}")
        kwargs[name] = value
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as e:
        raise SaveError(f"{label}.settings: {e}") from None


def _renamed(type_name, settings, label):
    """A copy of ``settings`` with the renamed parameters of the C++ table
    (include/screamer/serial/settings_compat.h) under their new names, each
    at its old position; a file that gives both names is refused."""
    for old, new in _b._serial_renames(type_name):
        if old not in settings:
            continue
        if new in settings:
            raise SaveError(f"{label}.settings.{old}: renamed to {new}, and the file gives both")
        settings = {(new if k == old else k): v for k, v in settings.items()}
    return settings


def _build_op(node, path):
    if not isinstance(node, dict) or set(node) != {"type", "settings"}:
        raise SaveError(f"{path}: expected {{'type': ..., 'settings': {{...}}}}")
    return _build(node["type"], node["settings"], path)


# ---- The methods of every operator ------------------------------------------

def _check(op):
    """Raise NotSaveable unless ``op`` saves as its own class."""
    name = type(op).__name__
    if not op._saveable():
        raise NotSaveable(f"{name}: the operator has no save and load support yet")
    if op._save_type() != name:
        raise NotSaveable(f"{name}: saves as {op._save_type()}, which is another operator")


def save(self, path):
    """Write the operator's settings and state to ``path``: CBOR for a
    ``.cbor`` path, JSON otherwise. ``screamer.load`` reads it back."""
    path = os.fspath(path)
    cbor = str(path).lower().endswith(".cbor")
    data = self.dumps(format="cbor" if cbor else "json")
    if cbor:
        with open(path, "wb") as fh:
            fh.write(data)
    else:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(data)


def dumps(self, format="json"):
    """The save file of the operator: ``str`` for ``format="json"``,
    ``bytes`` for ``format="cbor"``. ``screamer.loads`` reads it back."""
    if format not in ("json", "cbor"):
        raise ValueError(f"format must be 'json' or 'cbor', got {format!r}")
    _check(self)
    return self._dumps(format == "cbor")


def settings(self):
    """The constructor arguments, by name, as a dict; an optional argument
    that was not given is None."""
    _check(self)
    return json.loads(self._settings_json())


def state(self):
    """The state, ``{"version": N, "values": {...}}``."""
    _check(self)
    return json.loads(self._state_json())


def set_state(self, state):
    """Load a state that ``state()`` returned. A state that does not fit
    raises StateMismatch (or StateVersionError) and leaves the operator
    unchanged."""
    _check(self)
    self._set_state_json(json.dumps(state), type(self).__name__)


def install_methods(cls):
    """Add the save methods to an operator class, and the ``reset`` that also
    drops the per-column state of a resumed (T, K) batch."""
    for fn in (save, dumps, settings, state, set_state):
        setattr(cls, fn.__name__, fn)
    cls.reset = _b.EvalOp.reset
