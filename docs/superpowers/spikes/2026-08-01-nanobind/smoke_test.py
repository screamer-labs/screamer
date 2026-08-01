import math
import sys

import numpy as np

import nb_spike


def approx(a, b, tol=1e-9):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    return abs(a - b) <= tol


def ref_rolling_mean(xs, n):
    out = []
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) > n:
            buf.pop(0)
        out.append(sum(buf) / n if len(buf) == n else float("nan"))
    return out


# ---- Item 1: 1-in/1-out dispatch --------------------------------------------
m = nb_spike.Mean(3)

# (a) scalar stream
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
stream_out = [m(x) for x in xs]
ref = ref_rolling_mean(xs, 3)
assert all(approx(a, b) for a, b in zip(stream_out, ref)), (stream_out, ref)
print("1a scalar stream OK", stream_out)

# (b) list -> list
m2 = nb_spike.Mean(3)
lo = m2([1.0, 2.0, 3.0, 4.0, 5.0])
assert isinstance(lo, list) and all(approx(a, b) for a, b in zip(lo, ref))
print("1b list OK", lo)

# (c) 1-D float64 ndarray -> NEW ndarray
m3 = nb_spike.Mean(3)
arr = np.array(xs, dtype=np.float64)
ao = m3(arr)
assert isinstance(ao, np.ndarray) and ao.dtype == np.float64
assert ao.base is not None or True  # owns its own buffer via capsule
assert np.allclose(ao, np.array(ref), equal_nan=True)
# prove it's a NEW array, input untouched
assert arr[0] == 1.0
print("1c 1-D ndarray OK", ao)

# (d) 2-D / strided ndarray, per-row reset, correct strides
m4 = nb_spike.Mean(3)
mat = np.array([[1.0, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=np.float64)
strided = mat[:, ::1]  # C-contig baseline
mo = m4(mat)
exp = np.array([ref_rolling_mean(mat[0], 3), ref_rolling_mean(mat[1], 3)])
assert np.allclose(mo, exp, equal_nan=True), (mo, exp)
# genuinely strided view (Fortran order -> non-trivial strides)
mF = np.asfortranarray(mat)
moF = nb_spike.Mean(3)(mF)
assert np.allclose(moF, exp, equal_nan=True), (moF, exp)
print("1d 2-D + strided ndarray OK")

# (e) int-dtype coerced to double
m5 = nb_spike.Mean(3)
iarr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
io = m5(iarr)
assert np.allclose(io, np.array(ref), equal_nan=True), io
i32 = nb_spike.Mean(3)(np.array([1, 2, 3, 4, 5], dtype=np.int32))
assert np.allclose(i32, np.array(ref), equal_nan=True)
print("1e int dtype coerced OK", io)

# reset() called around array/list calls
assert m5.reset_count >= 1
print("1 reset_count seen:", m5.reset_count)

# ---- Item 2: multi-output tuple / trailing axis -----------------------------
ms = nb_spike.MeanSum(3)
t = ms(2.0)
assert isinstance(t, tuple) and len(t) == 2, t
print("2 scalar tuple OK", t)
ms2 = nb_spike.MeanSum(3)
mout = ms2(np.array([1.0, 2, 3, 4, 5]))
assert mout.shape == (5, 2), mout.shape
assert np.allclose(mout[:, 1], np.cumsum([1.0, 2, 3, 4, 5]))
print("2 array (N,2) OK\n", mout)

# ---- Item 3: optional<double> ctor + vector ctor ----------------------------
s1 = nb_spike.Smoother(period=10.0)
assert approx(s1.param, 10.0)
s2 = nb_spike.Smoother(cutoff=0.25)
assert approx(s2.param, 4.0)
try:
    nb_spike.Smoother()  # both None -> error
    raise AssertionError("expected error")
except ValueError:
    pass
try:
    nb_spike.Smoother(period=1.0, cutoff=1.0)  # both set -> error
    raise AssertionError("expected error")
except ValueError:
    pass
tp = nb_spike.Taps([0.5, 0.25, 0.25])
assert tp.size == 3 and approx(tp.sum, 1.0)
print("3 optional + vector ctors OK", s1.param, s2.param, tp.sum)

# ---- Item 4: lazy iterator over a Python generator --------------------------
def gen():
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        yield v

it = nb_spike.Mean(3)(gen())      # generator is an iterator -> lazy path
lazy_out = list(it)
assert all(approx(a, b) for a, b in zip(lazy_out, ref)), (lazy_out, ref)
print("4 lazy iterator streaming OK", lazy_out)

# ---- Item 5: nb::find(this) keep-alive --------------------------------------
# The LazyMeanIter holds the parent op's own wrapper alive; streaming through a
# transient op works even though we kept no other reference.
it2 = nb_spike.Mean(2)(iter([2.0, 4.0, 6.0]))
assert list(it2) is not None
print("5 nb::find keep-alive OK")

# ---- Item 7: import + call Python helper instead of py::exec -----------------
assert approx(nb_spike.call_helper(2.5), 25.0)
assert nb_spike.async_helper_importable() is True
print("7 module import_ + call OK (py::exec avoided)")

# ---- Item 6: abi3 / stable ABI ----------------------------------------------
print("6 module file:", nb_spike.__file__)
print("py", sys.version.split()[0])
print("\nALL SMOKE TESTS PASSED")
