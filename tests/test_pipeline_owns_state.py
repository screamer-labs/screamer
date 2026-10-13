"""Each Pipeline owns its operators' state (tracker H4).

A Pipeline compiles a copy of every operator it is built from (the same
parameters, a fresh state), so two Pipelines built from the same nodes, a
batch call beside a live session of another Pipeline, and direct calls of the
original operator never interfere. JavaScript: js/test/dag_lifetime.test.ts.
"""
import numpy as np

import screamer as sc
from screamer import Input, Pipeline


def test_two_pipelines_from_the_same_nodes_have_their_own_state():
    x = Input("x")
    y = sc.RollingSum(2)(x)
    p1, p2 = Pipeline([x], [y]), Pipeline([x], [y])
    s1, s2 = p1.live(), p2.live()
    for t in range(4):
        s1.push("x", t, 1.0)
        s2.push("x", t, 100.0)
    s1.flush()
    s2.flush()
    v1, _ = s1.result()
    v2, _ = s2.result()
    np.testing.assert_array_equal(v1, [np.nan, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(v2, [np.nan, 200.0, 200.0, 200.0])


def test_a_batch_call_does_not_reset_another_pipelines_session():
    x = Input("x")
    y = sc.RollingSum(3)(x)
    p1, p2 = Pipeline([x], [y]), Pipeline([x], [y])
    s2 = p2.live()
    s2.push("x", 0, 1.0).push("x", 1, 1.0)
    p1(np.array([5.0, 5.0, 5.0]))                 # resets p1's state only
    s2.push("x", 2, 1.0).flush()
    v, _ = s2.result()
    np.testing.assert_array_equal(v, [np.nan, np.nan, 3.0])


def test_the_original_operator_is_independent_of_the_pipeline():
    op = sc.RollingSum(2)
    x = Input("x")
    p = Pipeline([x], [op(x)])
    s = p.live()
    s.push("x", 0, 1.0)
    assert np.isnan(op(50.0))                      # the op's own state
    s.push("x", 1, 1.0).flush()
    v, _ = s.result()
    np.testing.assert_array_equal(v, [np.nan, 2.0])
    assert op(50.0) == 100.0                        # unaffected by the session
    del op                                          # the Pipeline keeps its copy
    np.testing.assert_array_equal(p(np.array([1.0, 2.0, 3.0]))[0], [np.nan, 3.0, 5.0])


def test_a_functor_reducer_is_owned_too():
    x = Input("x")
    agg = sc.ExpandingMean()
    node = sc.Resample(freq=2, agg=agg)(x)
    p1, p2 = Pipeline([x], [node]), Pipeline([x], [node])
    s1, s2 = p1.live(), p2.live()
    for t in range(4):
        s1.push("x", t, 1.0)
        s2.push("x", t, 10.0)
    s1.flush()
    s2.flush()
    np.testing.assert_array_equal(s1.result()[0], [1.0, 1.0])
    np.testing.assert_array_equal(s2.result()[0], [10.0, 10.0])


# Fix round 1, N3: every live session and lazy call of one Pipeline runs on an
# engine and operator copies of its own.

def test_two_sessions_of_one_pipeline_do_not_mix():
    x = Input("x")
    p = Pipeline([x], [sc.RollingMean(2)(x)])
    s3, s4 = p.live(), p.live()
    for t in range(3):
        s3.push("x", t, 1.0)
        s4.push("x", t, 100.0)
    s3.flush()
    s4.flush()
    v3, k3 = s3.result()
    v4, k4 = s4.result()
    assert list(k3) == list(k4) == [0, 1, 2]
    np.testing.assert_array_equal(v3, [np.nan, 1.0, 1.0])
    np.testing.assert_array_equal(v4, [np.nan, 100.0, 100.0])


def test_a_batch_call_and_a_lazy_run_beside_an_open_session():
    x = Input("x")
    p = Pipeline([x], [sc.RollingSum(2)(x)])
    s = p.live()
    s.push("x", 0, 1.0)
    p(np.array([7.0, 7.0, 7.0]))                   # a batch call: no flush of s
    lazy = p(iter([(5.0, 0), (5.0, 1)]))
    s.push("x", 1, 1.0)
    assert [r[0] for r in lazy][1] == 10.0          # the lazy run, unaffected
    s.push("x", 2, 1.0).flush()
    v, _ = s.result()
    np.testing.assert_array_equal(v, [np.nan, 2.0, 2.0])


def test_interleaved_lazy_runs_of_one_pipeline():
    x = Input("x")
    p = Pipeline([x], [sc.RollingSum(2)(x)])
    a = p(iter([(1.0, t) for t in range(4)]))
    b = p(iter([(100.0, t) for t in range(4)]))
    rows = [(next(a)[0], next(b)[0]) for _ in range(4)]
    np.testing.assert_array_equal([r[0] for r in rows], [np.nan, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal([r[1] for r in rows], [np.nan, 200.0, 200.0, 200.0])
