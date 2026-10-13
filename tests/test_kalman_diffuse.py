"""KalmanFilter: an infinite initial_variance is a diffuse prior (the first
observation sets the state); a finite variance whose arithmetic overflows
gives the 2.8.0 results, bit for bit (the diffuse path is decided by the
setting, not by an overflow)."""
import math

import numpy as np

import screamer as s


def test_a_finite_variance_that_overflows_gives_the_2_8_0_result():
    x = np.array([1.0, 2.0, 3.0])
    # Recorded with the 2.8.0 wheel.
    assert np.isnan(s.KalmanFilter(1.7e308, 0.5, 0.0, 1.7e308)(x)).all()
    assert np.isnan(s.KalmanFilter(1e308, 0.5, 0.0, 1e308)(x)).all()
    got = s.KalmanFilter(0.01, 1.0, 0.0, 1e300)(x).tolist()
    assert got == [1.0, 1.00990099009901, 1.0487331327055627]


def test_an_infinite_variance_is_a_diffuse_prior():
    x = np.array([np.nan, 4.0, 5.0, 6.0])
    k = s.KalmanFilter(0.01, 1.0, 0.0, math.inf)
    out = k(x)
    assert math.isnan(out[0]) and out[1] == 4.0
    # After the first observation the state has the observation variance.
    ref = s.KalmanFilter(0.01, 1.0, 4.0, 1.0)(x[2:])
    assert out[2:].tolist() == ref.tolist()
    assert k(x).tolist()[1:] == out.tolist()[1:]          # a batch call starts anew
