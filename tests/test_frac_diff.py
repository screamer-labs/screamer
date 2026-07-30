"""FracDiff: fractional differentiation (Lopez de Prado, AFML ch. 5).

The integer orders are the load-bearing checks. FracDiff's taps come from a
recursion; Diff and Diff2 are written independently of it, so agreement at
d = 1 and d = 2 pins the recursion against code that knows nothing about it.
"""
import numpy as np
import pytest
from scipy.special import binom

from screamer import FracDiff, Diff, Diff2, Input, Pipeline
from tests.regime_helpers import assert_batch_equals_scalar


def _walk(n=300, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _impulse(n=40):
    x = np.zeros(n)
    x[0] = 1.0
    return x


# ---------------------------------------------------------------------------
# Integer orders are exact identities against the rest of the library
# ---------------------------------------------------------------------------

def test_d0_is_identity():
    x = _walk()
    np.testing.assert_array_equal(FracDiff(d=0.0)(x), x)


def test_d1_equals_diff():
    x = _walk()
    np.testing.assert_array_equal(FracDiff(d=1.0, window_size=100)(x), Diff(1)(x))


def test_d2_equals_diff2():
    """Not bit-exact: FracDiff sums 1*x[t] - 2*x[t-1] + 1*x[t-2] in tap order
    while Diff2 chains two subtractions, so the two associate the same terms
    differently. The gap is a rounding step (~5e-15 relative)."""
    x = _walk()
    np.testing.assert_allclose(
        FracDiff(d=2.0, window_size=100)(x), Diff2()(x), equal_nan=True, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [0.1, 0.3, 0.5, 0.75, 0.99, 1.5, -0.4])
def test_impulse_response_is_the_binomial_weights(d):
    """Feeding a unit impulse reads the taps straight out of the filter:
    y[t] = sum_k w_k x[t-k] collapses to y[t] = w_t."""
    n = 40
    got = FracDiff(d=d, window_size=n, threshold=0.0, start_policy="zero")(_impulse(n))
    expected = np.array([(-1.0) ** k * binom(d, k) for k in range(n)])
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_weight_signs_and_decay_for_fractional_d():
    """At 0 < d < 1 the current sample carries weight 1 and every earlier
    sample a negative weight, with magnitudes decaying monotonically. The
    signs do not alternate the way they do at integer orders."""
    n = 60
    w = FracDiff(d=0.4, window_size=n, threshold=0.0, start_policy="zero")(_impulse(n))
    assert w[0] == 1.0
    assert np.all(w[1:] < 0)
    assert np.all(np.diff(np.abs(w[1:])) < 0)
    # The taps sum toward zero, which is what makes the filter remove a level.
    assert abs(w.sum()) < 0.2


# ---------------------------------------------------------------------------
# Truncation by threshold and window_size
# ---------------------------------------------------------------------------

def _tap_count(op, n=400):
    """The resolved tap count L, read off the strict warmup: the filter
    withholds output until L samples have arrived, so the first finite value
    sits at index L - 1."""
    out = op(_walk(n))
    return int(np.isnan(out).sum()) + 1


def test_threshold_above_first_weight_collapses_to_identity():
    x = _walk()
    # |w_1| = 0.4 at d = 0.4, so a threshold above it keeps only w_0.
    np.testing.assert_array_equal(FracDiff(d=0.4, threshold=0.5)(x), x)


def test_tighter_threshold_keeps_more_taps():
    loose = _tap_count(FracDiff(d=0.4, window_size=400, threshold=1e-2))
    tight = _tap_count(FracDiff(d=0.4, window_size=400, threshold=1e-6))
    assert loose < tight


def test_window_size_caps_the_tap_count():
    assert _tap_count(FracDiff(d=0.4, window_size=12, threshold=0.0)) == 12


def test_smaller_d_needs_more_taps_at_the_same_threshold():
    """Weights decay more slowly as d approaches 0."""
    assert (_tap_count(FracDiff(d=0.8, window_size=400))
            < _tap_count(FracDiff(d=0.2, window_size=400)))


# ---------------------------------------------------------------------------
# Warmup policies
# ---------------------------------------------------------------------------

def test_expanding_and_zero_coincide():
    """For a linear filter, using the samples available is the same
    arithmetic as padding the missing past with zeros."""
    x = _walk()
    np.testing.assert_array_equal(
        FracDiff(d=0.4, window_size=20, start_policy="expanding")(x),
        FracDiff(d=0.4, window_size=20, start_policy="zero")(x),
    )


def test_strict_matches_expanding_after_warmup():
    x = _walk()
    strict = FracDiff(d=0.4, window_size=20, start_policy="strict")(x)
    expanding = FracDiff(d=0.4, window_size=20, start_policy="expanding")(x)
    warm = np.isfinite(strict)
    np.testing.assert_allclose(strict[warm], expanding[warm], rtol=1e-12)
    assert not warm[0]                      # strict withholds the first sample
    assert np.isfinite(expanding[0])        # expanding does not


# ---------------------------------------------------------------------------
# NaN policy: ignore
# ---------------------------------------------------------------------------

def test_single_nan_costs_exactly_one_output():
    x = _walk(60)
    holed = x.copy()
    holed[30] = np.nan
    out = FracDiff(d=0.4, window_size=10)(holed)
    assert np.isnan(out[30])
    assert np.isfinite(out[31])


def test_nan_leaves_state_untouched():
    """The run with a NaN spliced in must equal the run without it, once the
    extra output slot is removed."""
    x = _walk(60)
    with_hole = np.insert(x, 30, np.nan)
    op = FracDiff(d=0.4, window_size=10)
    np.testing.assert_allclose(
        np.delete(op(with_hole), 30), op(x), equal_nan=True, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Every regime: eager, graph, lazy
# ---------------------------------------------------------------------------

def test_batch_equals_scalar_loop():
    assert_batch_equals_scalar(lambda: FracDiff(d=0.4, window_size=20), _walk())


def test_graph_regime_equals_batch():
    x = _walk(50)
    batch = FracDiff(d=0.4, window_size=20)(x)
    node = Input("x")
    dag = Pipeline(inputs=[node], outputs=[FracDiff(d=0.4, window_size=20)(node)])
    values, _index = dag((x, np.arange(len(x))))
    np.testing.assert_allclose(values, batch, equal_nan=True, rtol=1e-12)


def test_lazy_regime_equals_batch():
    x = _walk(50)
    batch = FracDiff(d=0.4, window_size=20)(x)
    node = Input("x")
    dag = Pipeline(inputs=[node], outputs=[FracDiff(d=0.4, window_size=20)(node)])
    rows = list(dag((float(v), int(i)) for i, v in enumerate(x)))
    np.testing.assert_allclose(
        np.array([r[0] for r in rows]), batch, equal_nan=True, rtol=1e-12
    )


def test_reset_restores_initial_state():
    x = _walk(50)
    op = FracDiff(d=0.4, window_size=20)
    first = np.array([op(v) for v in x])
    op.reset()
    second = np.array([op(v) for v in x])
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"window_size": 0},
    {"window_size": -5},
    {"threshold": -1e-6},
    {"start_policy": "sloppy"},
])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        FracDiff(**kwargs)


def test_negative_d_is_fractional_integration():
    """Weights share one sign and the filter still produces finite output."""
    n = 30
    w = FracDiff(d=-0.5, window_size=n, threshold=0.0, start_policy="zero")(_impulse(n))
    assert np.all(w > 0)
    assert np.all(np.isfinite(FracDiff(d=-0.5, window_size=20)(_walk())[19:]))


# ---------------------------------------------------------------------------
# The property the operator exists for
# ---------------------------------------------------------------------------

def test_partial_differencing_retains_more_memory_than_a_full_difference():
    """The docs claim d < 1 keeps a trace of the level. On a random walk the
    partially differenced series still correlates with the level; the first
    difference does not."""
    x = _walk(2000, seed=7)
    partial = FracDiff(d=0.4, window_size=200)(x)
    whole = FracDiff(d=1.0, window_size=200)(x)
    warm = np.isfinite(partial) & np.isfinite(whole)
    corr_partial = abs(np.corrcoef(x[warm], partial[warm])[0, 1])
    corr_whole = abs(np.corrcoef(x[warm], whole[warm])[0, 1])
    assert corr_partial > 0.5
    assert corr_whole < 0.1
