"""The Resample settings rule is one C++ check (dag/resample_check.h), shared
with the JavaScript binding: the messages below are the JavaScript ones too
(js/test/final_review.test.ts)."""

import numpy as np
import pytest

import screamer as s
from screamer import Input, Pipeline, Resample

X = np.arange(5.0)


@pytest.mark.parametrize("kw, msg", [
    (dict(every=0), "resample: every must be positive"),
    (dict(every=float("nan")), "resample: every must be positive"),
    (dict(every=float("inf")), "resample: every must be within the int64 range, got inf"),
    (dict(count=0), "resample: count must be >= 1"),
    (dict(count=float("inf")), "resample: count must be within the int64 range, got inf"),
    (dict(every=2, fill="bogus"), "resample: fill must be one of ('skip', 'nan', 'carry')"),
    (dict(every=2, label="center"), 'resample: label must be "left" or "right"'),
    (dict(every=2, label=None), 'resample: label must be "left" or "right"'),
    (dict(every=2, max_age=3), "resample: max_age requires clock=True, agg='last', and fill='carry'"),
    (dict(every=2, count=2), "resample: pass exactly one of freq=, every=, count=, threshold=, or clock="),
])
def test_value_errors(kw, msg):
    with pytest.raises(ValueError) as e:
        Resample(**kw)(X)
    assert str(e.value) == msg


def test_a_nan_threshold_is_refused():
    with pytest.raises(ValueError, match="threshold must be > 0"):
        Resample(X, np.ones(5), threshold=float("nan"))


def test_the_engine_refuses_a_zero_width_from_a_description():
    x = Input("x")
    d = Pipeline([x], [Resample(every=2)(x)]).to_dict()
    d["nodes"][1]["params"]["every"] = 0
    with pytest.raises(ValueError, match="every must be positive"):
        Pipeline.from_dict(d)


def test_backtest_report_index_has_one_value_per_row():
    with pytest.raises(ValueError) as e:
        s.backtest_report(np.zeros((5, 4)), index=[0, 1])
    assert str(e.value) == "backtest_report: the stream has 5 rows of values but 2 index values"
    running, _ = s.backtest_report(np.zeros((5, 4)), index=np.arange(5))
    assert len(running["index"]) == 5
