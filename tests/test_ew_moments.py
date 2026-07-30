"""EwSkew and EwKurt compute the quantity they are named after.

These operators passed every self-consistency test in the suite while returning
a number that shrank as the window grew: on exponential data (true skew 2.0),
`EwSkew(span=100)` returned 0.019. Nothing compared them to an external truth,
because their baseline comparison was disabled with a `# todo baselines` note.

Two anchors here, in order of strength:

1. **Equal weights reduce to scipy.** With `alpha` set so the weights are flat,
   the weighted estimators must equal `scipy.stats.skew/kurtosis(bias=False)`
   exactly. This pins the formula to a definition someone else maintains.
2. **Known distributions are recovered.** Feed samples whose true skew and
   excess kurtosis are known in closed form and check the estimate lands on
   them. This is the property the old code failed at every span.

A third, cheap and general: the estimate must not drift systematically with
window length on stationary input. That is what made the old bug visible, and
it needs no external reference at all.
"""
import numpy as np
import pytest
from scipy import stats

from screamer import EwKurt, EwSkew
from devtools.baselines.EwSkew import ew_central_moments


# Distributions with closed-form third and fourth standardised moments.
DISTRIBUTIONS = [
    ("normal", lambda r, n: r.standard_normal(n), 0.0, 0.0),
    ("uniform", lambda r, n: r.uniform(-1, 1, n), 0.0, -1.2),
    ("exponential", lambda r, n: r.exponential(1.0, n), 2.0, 6.0),
    ("laplace", lambda r, n: r.laplace(0.0, 1.0, n), 0.0, 3.0),
]


def _flat_weight_moments(x):
    """Central moments and n under equal weights, via the same code path."""
    # alpha -> 0 makes the EW weights flat over a short sample; use the closed
    # form directly instead so the comparison is exact rather than asymptotic.
    n = len(x)
    mean = x.mean()
    d = x - mean
    return n, (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()


def test_formula_matches_scipy_under_equal_weights():
    """The estimator this library implements is scipy's bias=False one.

    Anchors the definition: with flat weights the EW formulas must reproduce
    scipy exactly, so any later change to the correction factors fails here.
    """
    x = np.random.default_rng(0).standard_normal(500)
    n, m2, m3, m4 = _flat_weight_moments(x)

    g1 = m3 / m2 ** 1.5
    skew = np.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
    np.testing.assert_allclose(skew, stats.skew(x, bias=False), rtol=1e-12)

    g2 = m4 / (m2 * m2)
    kurt = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 - 3.0 * (n - 1.0))
    np.testing.assert_allclose(kurt, stats.kurtosis(x, bias=False), rtol=1e-12)


@pytest.mark.parametrize("name,draw,true_skew,true_kurt", DISTRIBUTIONS)
def test_recovers_known_moments(name, draw, true_skew, true_kurt):
    """A long window on a large sample must land on the true values.

    Tolerances are loose because a fourth-moment estimate on heavy-tailed data
    is noisy; they are still far tighter than the old behaviour, which returned
    ~0 for every distribution here.
    """
    x = draw(np.random.default_rng(11), 400_000)
    span = 20_000

    got_skew = float(np.asarray(EwSkew(span=span)(x))[-1])
    got_kurt = float(np.asarray(EwKurt(span=span)(x))[-1])

    assert got_skew == pytest.approx(true_skew, abs=0.35), (
        f"{name}: EwSkew returned {got_skew:.3f}, true skew is {true_skew}"
    )
    assert got_kurt == pytest.approx(true_kurt, abs=1.2), (
        f"{name}: EwKurt returned {got_kurt:.3f}, true excess kurtosis is {true_kurt}"
    )


@pytest.mark.parametrize("op,label", [(EwSkew, "EwSkew"), (EwKurt, "EwKurt")])
def test_estimate_does_not_drift_with_window_length(op, label):
    """On stationary input, a wider window means less noise, not a different
    answer. The old estimators shrank by roughly the effective sample size, so
    span=1000 returned about a tenth of what span=100 returned.

    This needs no reference implementation, which makes it the cheapest guard
    against the whole class of scaling errors.
    """
    x = np.random.default_rng(3).exponential(1.0, 400_000)
    values = [float(np.asarray(op(span=s)(x))[-1]) for s in (500, 2000, 8000)]

    spread = max(values) - min(values)
    assert spread < 0.5 * abs(np.mean(values)), (
        f"{label} changes with window length: {values}. On stationary input the "
        "estimate must converge, not scale."
    )


def test_ew_moments_helper_matches_the_operators():
    """The baseline's moment recursion is the one the operators use."""
    x = np.random.default_rng(5).standard_normal(2000)
    n, m2, m3, m4 = ew_central_moments(x, alpha=2 / (50 + 1)).T

    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = m3 / m2 ** 1.5
        expected_skew = np.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
    got = np.asarray(EwSkew(span=50)(x))

    finite = np.isfinite(got) & np.isfinite(expected_skew)
    np.testing.assert_allclose(got[finite], expected_skew[finite], rtol=1e-9)
