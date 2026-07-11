import numpy as np
from screamer.dag import Node, Input, is_node
from screamer import combine_latest, RollingCorr


def test_input_is_a_node():
    a = Input("price_a")
    assert is_node(a)
    assert a.op == ("input", "price_a")
    assert a.inputs == ()


def test_combinator_on_nodes_builds_node():
    a, b = Input("a"), Input("b")
    n = combine_latest(a, b, func=None)
    assert is_node(n)
    assert n.inputs == (a, b)
    assert n.op[0] == "operator"
    # func= is not forwarded to the node kwargs (not a graph-level concept);
    # only emit= is stored.
    assert n.op[2] == {"emit": "when_all"}


def test_stream_operator_on_data_still_computes():
    # no Node args -> normal eager behavior, returns arrays not a Node
    a_k = np.array([1, 2, 3], dtype=np.int64)
    a_v = np.array([1.0, 2.0, 3.0])
    b_k = np.array([1, 2, 3], dtype=np.int64)
    b_v = np.array([4.0, 5.0, 6.0])
    aligned, keys = combine_latest(a_v, b_v, index=[a_k, b_k])
    assert not is_node(aligned)
    assert aligned.shape == (3, 2)  # 3 distinct index values (coalesced, one row per index)


def test_functor_hook_single_node():
    a = Input("a")
    n = RollingCorr(20)(a)              # one Node arg -> Node
    assert is_node(n)
    assert n.inputs == (a,)


def test_functor_hook_multiple_nodes():
    a, b = Input("a"), Input("b")
    n = RollingCorr(20)(a, b)           # two Node args -> Node with two inputs
    assert is_node(n)
    assert n.inputs == (a, b)


def test_combine_latest_func_on_nodes_raises():
    import pytest
    a, b = Input("a"), Input("b")
    with pytest.raises(ValueError):
        combine_latest(a, b, func=lambda p, q: p - q)   # func not allowed in a graph
    # func=None on nodes is still valid (alignment-only) — should NOT raise:
    n = combine_latest(a, b, func=None)
    assert is_node(n)
