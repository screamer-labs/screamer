import screamer
from screamer import dag as dag_mod


def test_dag_names_exported():
    for name in ("Node", "Input", "Dag"):
        assert hasattr(screamer, name)
        assert getattr(screamer, name) is getattr(dag_mod, name)
        assert name in screamer.__all__
