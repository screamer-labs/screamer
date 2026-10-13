"""The bench_pipeline gate fails on every kind of failure (devtools/bench_pipeline.py selftest)."""
import importlib.util
import pathlib


def test_bench_pipeline_gate_selftest(capsys):
    path = pathlib.Path(__file__).resolve().parents[1] / "devtools" / "bench_pipeline.py"
    spec = importlib.util.spec_from_file_location("bench_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.selftest() == 0, capsys.readouterr().out
