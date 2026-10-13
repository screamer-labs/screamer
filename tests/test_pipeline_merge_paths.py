"""A batch Pipeline merges up to 8 inputs by a scan and more by a heap. Both
merges order events by index, the lower input first on equal indexes, as a
live session fed in that order does, and both name the input and the event
whose index decreases."""
import numpy as np
import pytest

import screamer as s


def graph(n, emit):
    ins = [s.Input(f"x{i}") for i in range(n)]
    joined = s.CombineLatest(emit=emit)(*ins)
    return s.Pipeline(ins, [joined, s.Lag(1)(ins[-1])]), [f"x{i}" for i in range(n)]


def feeds(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = int(rng.integers(5, 40))
        idx = np.sort(rng.integers(0, 25, m)).astype(np.int64)    # many equal indexes
        out.append((rng.standard_normal(m), idx))
    return out


def same(a, b):
    return all(np.asarray(x).tobytes() == np.asarray(y).tobytes() and
               np.shape(x) == np.shape(y) for x, y in zip(a, b))


@pytest.mark.parametrize("n", [2, 8, 9, 12])
@pytest.mark.parametrize("emit", ["when_all", "on_any"])
@pytest.mark.parametrize("seed", range(4))
def test_batch_merge_equals_live_in_merge_order(n, emit, seed):
    p, names = graph(n, emit)
    f = feeds(n, seed)
    batch = p(*f)
    events = sorted(((int(i[r]), k, r) for k, (v, i) in enumerate(f) for r in range(len(i))))
    live = p.live()
    for t, k, r in events:
        live.push(names[k], t, float(f[k][0][r]))
    live.flush()
    streamed = live.result()
    for b, l in zip(batch, streamed):
        assert same(b, l)


@pytest.mark.parametrize("n", [8, 9, 12])
def test_batch_merge_names_the_decreasing_input(n):
    p, _ = graph(n, "when_all")
    f = [(np.arange(6.0), np.arange(6, dtype=np.int64)) for _ in range(n)]
    f[5] = (np.arange(6.0), np.array([0, 1, 2, 3, 1, 5], dtype=np.int64))
    with pytest.raises(ValueError, match="the index of input 5 decreases at event 4, from 3 to 1"):
        p(*f)
