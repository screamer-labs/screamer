"""An every=/count= Resample forwards a watermark at its open bar as events
arrive (fix round 1, N1).

The open bar emits no row below its first index (its start for every=), so
the Resample says so on each new bar. A live output alignment beside it, or a
join fed by it, then holds at most the rows of one open bar.
"""
import numpy as np
import pytest

import screamer as sc
from screamer import Input, Pipeline
from screamer.dag import make_operator_node


@pytest.mark.parametrize("label", ["left", "right"])
@pytest.mark.parametrize("mode", ["every", "count"])
def test_a_long_bar_beside_a_fast_output_holds_one_bar(mode, label):
    bar, n = 1000, 20_000
    x = Input("x")
    kw = {"every": bar} if mode == "every" else {"count": bar}
    p = Pipeline([x], [sc.Resample(agg="last", label=label, **kw)(x), sc.RollingMean(1)(x)])
    want = np.asarray(p(np.arange(float(n)))[0][1])
    s = p.live()
    released = 0
    for t in range(n):
        s.push("x", t, float(t))
        if t % 100 == 99:
            (_, k), _ = s.result()
            released += len(k)
            due = np.searchsorted(want, t, side="right")    # batch rows up to t
            assert due - released <= bar + 100              # at most one open bar waits
    s.flush()
    (_, k), _ = s.result()
    assert released + len(k) == len(want)


def test_a_join_fed_by_a_long_bar_runs_in_batch_and_lazy():
    # A Dropna with two producers, one a Resample of 600-event bars: the join
    # holds one open bar (600 < max_pending), in batch, lazy and live.
    x = Input("x")
    node = make_operator_node(sc.Dropna, (x, sc.Resample(every=600, agg="last")(x)), {"how": "any"})
    p = Pipeline([x], [node], max_pending=1000)
    v, k = p(np.arange(10_000.0))
    assert len(k) == 10_000 + 17                  # 17 bars beside the 10 000 events
    lz = list(p(iter((float(t), t) for t in range(10_000))))
    assert [r[1] for r in lz] == list(k)
