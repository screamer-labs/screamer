"""Write wasm/smoke/null_oracle.json: for every operator setting, whether the
Python binding takes None for it (as "not given") or raises TypeError, and
whether it takes a bool (True) or raises TypeError, and that TypeError's
message ("RollingMean: window_size must be an integer, got bool"). For every
float setting it also records, for inf, -inf and NaN, the message of the
refusal ("RollingMean: ... must be finite, got inf", include/screamer/common/
finite_settings.h) or null when the setting takes the value.

js/test/ops.test.ts passes null, and then true, for each setting and checks
that JavaScript agrees. Each op is built from ``base``: its required settings filled in, and
for ops that need one of several settings (the Ew family), one of them. A
setting is tested with every other setting at its default.

Modes:
    (no args)  write the fixture
    --stdout   print the fixture to stdout (freshness gate)
"""
import argparse
import json
import os
import sys

import screamer as s

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "wasm_manifest.json")
OUT = os.path.join(HERE, "..", "..", "wasm", "smoke", "null_oracle.json")

# Values for settings that an op cannot be built without.
FILL = {"n": 3, "lower": 1.0, "upper": 3.0, "threshold": 1.0}


def _builds(cls, kw):
    try:
        cls(**kw)
        return True
    except Exception:
        return False


def _base(cls, args):
    """Keyword arguments that build the op, beyond the defaults."""
    kw = {a["name"]: FILL[a["name"]] for a in args
          if a["default"] is None and a["name"] in FILL}
    if _builds(cls, kw):
        return kw
    # One of several optional settings must be given (com/span/...).
    for a in args:
        for v in (10.0, 0.5):
            trial = dict(kw, **{a["name"]: v})
            if _builds(cls, trial):
                return trial
    for a in args:  # two related periods (RoofingFilter)
        for b in args:
            if a is not b:
                trial = dict(kw, **{a["name"]: 48.0, b["name"]: 10.0})
                if _builds(cls, trial):
                    return trial
    raise SystemExit(f"cannot build {cls.__name__}")


def _accepts(cls, base, name, value):
    """Whether the op takes `value` for setting `name`: False for a TypeError.
    Another error means the value was taken and the op refused the rest. A
    bool must be refused as a bool ("<Op>: <setting> must be ..., got bool"):
    a TypeError for another reason (a stale `base`) stops the generator."""
    try:
        cls(**dict(base, **{name: value}))
    except TypeError as e:
        if value is True and not str(e).startswith(f"{cls.__name__}: {name} must be ") \
                or value is True and not str(e).endswith(", got bool"):
            raise SystemExit(f"{cls.__name__}({name}=True): unexpected TypeError: {e}")
        return False
    except Exception:
        pass
    return True


def _message(cls, base, name, value):
    """The TypeError message for `value` as setting `name`, or None."""
    try:
        cls(**dict(base, **{name: value}))
    except TypeError as e:
        return str(e)
    except Exception:
        pass
    return None


# The decay settings of the Ew family: one of them is given at a time.
EW = {"com", "span", "halflife", "alpha"}
FLOAT_TYPES = ("double", "std::optional<double>", "std::vector<double>")
NON_FINITE = {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}


def _finite_message(cls, base, name, value):
    """The finite-setting refusal of `value` for setting `name`, or None when
    the setting takes it (the op may still refuse another setting)."""
    if name in EW:
        base = {k: v for k, v in base.items() if k not in EW}
    try:
        cls(**dict(base, **{name: value}))
    except ValueError as e:
        if str(e).startswith(f"{cls.__name__}: {name} must be finite"):
            return str(e)
    except Exception:
        pass
    return None


def _non_finite(cls, base, op):
    out = {}
    for cpp_type, arg in zip(op["ctor"], op["ctor_args"]):
        if cpp_type not in FLOAT_TYPES:
            continue
        vector = cpp_type == "std::vector<double>"
        out[arg["name"]] = {k: _finite_message(cls, base, arg["name"], [v] if vector else v)
                            for k, v in NON_FINITE.items()}
    return out


def _below_min(cls, base, op):
    """For every integer setting, the refusal of the value one below its
    smallest (C++ int_settings.h): "<Op>: <setting> must be at least 1, got 0"."""
    from screamer import screamer_bindings as _b
    out = {}
    for cpp_type, arg in zip(op["ctor"], op["ctor_args"]):
        if cpp_type != "int":
            continue
        low = _b._int_setting_min(op["name"], arg["name"])
        if low is None:
            raise SystemExit(f"{op['name']}.{arg['name']}: no entry in int_settings.h")
        try:
            cls(**dict(base, **{arg["name"]: low - 1}))
            out[arg["name"]] = None
        except ValueError as e:
            out[arg["name"]] = [low - 1, str(e)]
    return out


def compute():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    out = []
    for op in manifest:
        args = op["ctor_args"]
        if not args:
            continue
        cls = getattr(s, op["name"])
        base = _base(cls, args)
        settings = {a["name"]: _accepts(cls, base, a["name"], None) for a in args}
        bools = {a["name"]: _accepts(cls, base, a["name"], True) for a in args}
        out.append({"name": op["name"], "base": base, "accepts_none": settings,
                    "accepts_bool": bools,
                    "bool_message": {a["name"]: _message(cls, base, a["name"], True) for a in args},
                    "non_finite": _non_finite(cls, base, op),
                    "below_min": _below_min(cls, base, op)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true", help="print the fixture to stdout")
    args = ap.parse_args()
    text = json.dumps(compute(), indent=1, sort_keys=True) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
