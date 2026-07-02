import numpy as np
from screamer import streams


def _reference_merge(series):
    # Stable global sort by key, ties broken by source index (the series order).
    keys = np.concatenate([np.asarray(k) for k, _ in series])
    vals = np.concatenate([np.asarray(v) for _, v in series])
    src = np.concatenate([np.full(len(k), i, dtype=np.uint32)
                          for i, (k, _) in enumerate(series)])
    order = np.argsort(keys, kind="stable")   # stable => source-order tie-break
    return keys[order], vals[order], src[order]


def test_merge_two_int_sorted_series():
    a_k = np.array([1, 3, 5, 7], dtype=np.int64)
    a_v = np.array([10.0, 30.0, 50.0, 70.0])
    b_k = np.array([2, 4, 6], dtype=np.int64)
    b_v = np.array([20.0, 40.0, 60.0])
    got_k, got_v, got_s = streams.merge((a_k, a_v), (b_k, b_v))
    exp_k, exp_v, exp_s = _reference_merge([(a_k, a_v), (b_k, b_v)])
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_v, exp_v)
    np.testing.assert_array_equal(got_s, exp_s)


def test_merge_ties_break_by_source_order():
    a_k = np.array([1, 2, 2], dtype=np.int64)
    a_v = np.array([1.0, 2.0, 2.5])
    b_k = np.array([2, 3], dtype=np.int64)
    b_v = np.array([20.0, 30.0])
    got_k, got_v, got_s = streams.merge((a_k, a_v), (b_k, b_v))
    # at key==2: source 0's events (2.0, 2.5) come before source 1's (20.0)
    np.testing.assert_array_equal(got_k, np.array([1, 2, 2, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(got_v, np.array([1.0, 2.0, 2.5, 20.0, 30.0]))
    np.testing.assert_array_equal(got_s, np.array([0, 0, 0, 1, 1], dtype=np.uint32))


def test_merge_float_keys():
    a_k = np.array([0.5, 2.5], dtype=np.float64)
    a_v = np.array([5.0, 25.0])
    b_k = np.array([1.5], dtype=np.float64)
    b_v = np.array([15.0])
    got_k, got_v, got_s = streams.merge((a_k, a_v), (b_k, b_v))
    exp_k, exp_v, exp_s = _reference_merge([(a_k, a_v), (b_k, b_v)])
    np.testing.assert_array_equal(got_k, exp_k)
    np.testing.assert_array_equal(got_v, exp_v)
    np.testing.assert_array_equal(got_s, exp_s)


def test_pull_iter_matches_batch_identity():
    rng = np.random.default_rng(7)
    a_k = np.sort(rng.integers(0, 1000, size=200)).astype(np.int64)
    a_v = rng.standard_normal(200)
    b_k = np.sort(rng.integers(0, 1000, size=150)).astype(np.int64)
    b_v = rng.standard_normal(150)

    bk, bv, bs = streams.merge((a_k, a_v), (b_k, b_v))              # batch
    events = list(streams.merge_iter((a_k, a_v), (b_k, b_v)))       # streaming pull

    got_k = np.array([e[0] for e in events], dtype=np.int64)
    got_v = np.array([e[1] for e in events])
    got_s = np.array([e[2] for e in events], dtype=np.uint32)
    np.testing.assert_array_equal(got_k, bk)
    np.testing.assert_array_equal(got_v, bv)
    np.testing.assert_array_equal(got_s, bs)


import asyncio


def _drain(agen):
    async def run():
        out = []
        async for e in agen:
            out.append(e)
        return out
    return asyncio.run(run())


def test_pace_preserves_order_and_scales_sleeps():
    a_k = np.array([0, 10, 30], dtype=np.int64)     # deltas 10, 20
    a_v = np.array([0.0, 1.0, 3.0])
    b_k = np.array([5, 20], dtype=np.int64)          # interleaves -> 0,5,10,20,30
    b_v = np.array([0.5, 2.0])

    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    events = _drain(streams.pace((a_k, a_v), (b_k, b_v), speed=2.0, sleep=fake_sleep))

    # Order identical to a plain merge
    bk, bv, bs = streams.merge((a_k, a_v), (b_k, b_v))
    np.testing.assert_array_equal(np.array([e[0] for e in events], dtype=np.int64), bk)
    np.testing.assert_array_equal(np.array([e[1] for e in events]), bv)

    # Sleeps == successive key deltas / speed (first event has no preceding sleep).
    # merged keys: 0,5,10,20,30 -> deltas 5,5,10,10 -> /2.0 -> 2.5,2.5,5,5
    assert slept == [2.5, 2.5, 5.0, 5.0]


def test_pace_infinite_speed_no_sleep():
    a_k = np.array([0, 100], dtype=np.int64)
    a_v = np.array([0.0, 1.0])
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    events = _drain(streams.pace((a_k, a_v), speed=float("inf"), sleep=fake_sleep))
    assert [e[0] for e in events] == [0, 100]
    assert slept == []          # no pacing at infinite speed
