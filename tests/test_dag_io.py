"""Tests for DAG serialization (screamer/dag_io.py).

The backbone is round-trip equivalence: a graph -> JSON -> graph must run
identically. Also covers the schema shape and load-time error handling.
"""
import json

import numpy as np
import pytest

from screamer import RollingMean, EwMean, RollingMinMax, Sub, Input, Dag
from screamer.streams import combine_latest, dropna, resample, Resample, select


def _rich_dag(align_outputs):
    a, b = Input("a"), Input("b")
    cl = combine_latest(a, b)                     # 2-column shared node -> diamond
    sm = RollingMean(3)(dropna(Sub()(cl)))        # operator + functor + operator + functor
    rs = resample(sm, every=2, agg="last")         # resample operator (node-mode span)
    sel = select(cl, columns=[0])                 # select operator, second consumer of cl
    return Dag([a, b], [rs, sel], align_outputs=align_outputs)


def _feeds():
    fa = (np.array([10.0, 20.0, 30.0, 40.0, 50.0]), np.array([1, 2, 3, 4, 5]))
    fb = (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1, 2, 3, 4, 5]))
    return fa, fb


def _equal(r0, r1):
    if isinstance(r0[0], np.ndarray):  # single output: (values, index)
        r0, r1 = (r0,), (r1,)
    return all(np.array_equal(np.asarray(x[0]), np.asarray(y[0]), equal_nan=True)
               and np.array_equal(np.asarray(x[1]), np.asarray(y[1]))
               for x, y in zip(r0, r1))


@pytest.mark.parametrize("align", [True, False])
def test_round_trip_identical(align):
    dag = _rich_dag(align)
    rebuilt = Dag.from_json(dag.to_json())
    fa, fb = _feeds()
    assert _equal(dag(fa, fb), rebuilt(fa, fb))


@pytest.mark.parametrize("align", [True, False])
def test_round_trip_via_dict(align):
    dag = _rich_dag(align)
    rebuilt = Dag.from_dict(dag.to_dict())
    fa, fb = _feeds()
    assert _equal(dag(fa, fb), rebuilt(fa, fb))


def test_single_output_round_trip():
    a, b = Input("a"), Input("b")
    dag = Dag([a, b], [Sub()(combine_latest(a, b))])
    rebuilt = Dag.from_json(dag.to_json())
    fa, fb = _feeds()
    np.testing.assert_array_equal(np.asarray(dag(fa, fb)[0]),
                                  np.asarray(rebuilt(fa, fb)[0]))


def test_functor_params_preserved_through_json():
    a = Input("a")
    dag = Dag([a], [RollingMean(7)(a)])
    node = [n for n in dag.to_dict()["nodes"] if n["kind"] == "functor"][0]
    assert node["cls"] == "RollingMean"
    assert node["params"] == {"window_size": 7}
    rebuilt = Dag.from_json(dag.to_json())
    x = np.arange(20.0)
    np.testing.assert_allclose(np.asarray(dag(x)[0]), np.asarray(rebuilt(x)[0]), equal_nan=True)


def test_schema_shape():
    d = _rich_dag(True).to_dict()
    assert d["screamer_dag"] == 1
    assert d["inputs"] == ["a", "b"]
    assert d["align_outputs"] is True
    assert isinstance(d["nodes"], list) and isinstance(d["outputs"], list)
    # every edge references an earlier node id
    for node in d["nodes"]:
        for i in node.get("in", []):
            assert i < node["id"]


def test_input_order_preserved():
    x, y, z = Input("x"), Input("y"), Input("z")
    # declare in a specific order; outputs discover them in another
    dag = Dag([x, y, z], [Sub()(combine_latest(z, x)), EwMean(span=3)(y)])
    assert Dag.from_json(dag.to_json())._names == ["x", "y", "z"]


def test_bad_version_raises():
    d = _rich_dag(True).to_dict()
    d["screamer_dag"] = 999
    with pytest.raises(ValueError, match="version"):
        Dag.from_dict(d)


def test_unknown_functor_raises():
    d = _rich_dag(True).to_dict()
    for node in d["nodes"]:
        if node["kind"] == "functor":
            node["cls"] = "NoSuchFunctor"
            break
    with pytest.raises(ValueError, match="unknown functor"):
        Dag.from_dict(d)


def test_unknown_operator_raises():
    d = _rich_dag(True).to_dict()
    for node in d["nodes"]:
        if node["kind"] == "operator":
            node["op"] = "no_such_operator"
            break
    with pytest.raises(ValueError, match="unknown DAG operator"):
        Dag.from_dict(d)


def test_from_dict_does_not_mutate_input():
    import copy
    d = _rich_dag(True).to_dict()
    snapshot = copy.deepcopy(d)
    Dag.from_dict(d)
    assert d == snapshot


def test_numpy_scalar_params_serialize_and_round_trip():
    # A numpy scalar arg (common: RollingMean(arr.size // 10)) must serialize.
    a = Input("a")
    dag = Dag([a], [RollingMean(np.int64(20))(a)])
    d = dag.to_dict()
    assert type(d["nodes"][-1]["params"]["window_size"]) is int  # coerced to native
    json.dumps(d)  # must not raise
    rebuilt = Dag.from_json(dag.to_json())
    x = np.arange(40.0)
    np.testing.assert_allclose(np.asarray(dag(x)[0]), np.asarray(rebuilt(x)[0]), equal_nan=True)


def test_multi_output_functor_round_trip():
    # RollingMinMax is a 1-input, 2-output functor.
    a = Input("a")
    dag = Dag([a], [RollingMinMax(5)(a)])
    x = np.arange(30.0)
    rebuilt = Dag.from_json(dag.to_json())
    np.testing.assert_array_equal(np.asarray(dag(x)[0]), np.asarray(rebuilt(x)[0]))


def test_select_numpy_columns_round_trip():
    a, b = Input("a"), Input("b")
    dag = Dag([a, b], [select(combine_latest(a, b), columns=np.array([0]))])
    json.dumps(dag.to_dict())  # numpy array columns coerced to a list
    fa, fb = _feeds()
    assert _equal(dag(fa, fb), Dag.from_json(dag.to_json())(fa, fb))


def test_from_dict_positional_args_fallback():
    # A functor serialized in the schema-less positional form reconstructs.
    a = Input("a")
    d = Dag([a], [RollingMean(20)(a)]).to_dict()
    for node in d["nodes"]:
        if node["kind"] == "functor":
            node["params"] = {"args": [20]}
    rebuilt = Dag.from_dict(d)
    x = np.arange(30.0)
    np.testing.assert_allclose(np.asarray(rebuilt(x)[0]),
                               np.asarray(RollingMean(20)(x)), equal_nan=True)


def test_select_is_public():
    import screamer
    assert screamer.select is select
