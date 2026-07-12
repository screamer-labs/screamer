"""Tests for DAG visualization (screamer/dag_viz.py): the shared graph walk,
the ASCII tree, DOT output, and the Jupyter repr.
"""
import sys

import numpy as np
import pytest

from screamer import RollingMean, EwMean, Sub, Input, Pipeline
from screamer.streams import CombineLatest
from screamer.dag_viz import build_graph, to_text, to_dot


def _simple():
    a, b = Input("a"), Input("b")
    return Pipeline([a, b], [Sub()(CombineLatest()(a, b))])


def _diamond():
    # combine_latest is shared by two outputs -> a diamond, two outputs.
    a, b = Input("a"), Input("b")
    cl = CombineLatest()(a, b)
    return Pipeline([a, b], [RollingMean(20)(Sub()(cl)), EwMean(span=10)(cl)])


def test_build_graph_kinds_and_counts():
    nodes, input_ids, output_ids = build_graph(_simple())
    kinds = [n.kind for n in nodes]
    assert kinds.count("input") == 2
    assert kinds.count("operator") == 1
    assert kinds.count("functor") == 1
    assert len(input_ids) == 2 and len(output_ids) == 1


def test_build_graph_is_in_dependency_order():
    # every node's inputs were numbered before it
    nodes, _, _ = build_graph(_diamond())
    for n in nodes:
        for cid in n.in_ids:
            assert cid < n.id


def test_shared_node_deduplicated():
    nodes, _, output_ids = build_graph(_diamond())
    # the shared CombineLatest appears exactly once
    operators = [n for n in nodes if n.kind == "operator" and n.name == "CombineLatest"]
    assert len(operators) == 1
    # and it feeds two different consumers
    cl_id = operators[0].id
    consumers = [n.id for n in nodes if cl_id in n.in_ids]
    assert len(consumers) == 2


def test_functor_and_operator_labels_carry_params():
    dot = to_dot(_diamond())
    assert "RollingMean(window_size=20)" in dot
    assert "EwMean(span=10)" in dot
    assert "CombineLatest(emit='when_all')" in dot  # None-valued params dropped
    assert "func=None" not in dot


def test_to_text_header_and_outputs():
    text = to_text(_diamond())
    assert text.startswith("Pipeline(2 input(s), 2 output(s)")
    assert "out[0] =" in text and "out[1] =" in text


def test_to_text_references_shared_node():
    text = to_text(_diamond())
    # the shared node is expanded once and referenced (^#) the second time
    assert "^#" in text
    assert text.count("CombineLatest(emit=") == 1


def test_to_text_marks_inputs():
    text = to_text(_simple())
    assert "a  #0 (input)" in text or "a  #1 (input)" in text


def test_to_text_uses_box_drawing():
    text = to_text(_diamond())
    assert "└─" in text and "├─" in text  # Unicode corners, not ASCII fallback


def test_to_dot_has_all_nodes_and_edges():
    dag = _simple()
    nodes, _, _ = build_graph(dag)
    dot = to_dot(dag)
    assert dot.startswith("digraph screamer_dag")
    for n in nodes:
        assert f"n{n.id} [" in dot
    n_edges = sum(len(n.in_ids) for n in nodes)
    assert dot.count(" -> ") == n_edges


def test_to_dot_styles_inputs_and_outputs():
    dot = to_dot(_simple())
    assert "shape=ellipse" in dot          # inputs
    assert "penwidth=2" in dot             # outputs emphasized


def test_repr_is_concise():
    assert repr(_diamond()) == "Pipeline(2 input(s), 2 output(s))"


def test_str_equals_to_text():
    dag = _diamond()
    assert str(dag) == dag.to_text()


def test_mimebundle_always_has_text():
    dag = _simple()
    bundle = dag._repr_mimebundle_()
    assert bundle["text/plain"] == dag.to_text()


def test_to_graphviz_returns_source_when_available():
    graphviz = pytest.importorskip("graphviz")
    src = _simple().to_graphviz()
    assert isinstance(src, graphviz.Source)
    assert "digraph screamer_dag" in src.source


def test_to_graphviz_without_package_raises_helpful_error(monkeypatch):
    # Simulate graphviz being absent (None in sys.modules makes ``import
    # graphviz`` raise ImportError) so the helpful-error path is always
    # exercised, whether or not graphviz is installed in the environment.
    monkeypatch.setitem(sys.modules, "graphviz", None)
    with pytest.raises(ImportError, match="graphviz"):
        _simple().to_graphviz()
