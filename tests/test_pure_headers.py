import subprocess, pathlib
def test_operator_headers_compile_without_binding_library():
    root = pathlib.Path(__file__).resolve().parent.parent
    script = root / "devtools" / "check_pure_headers.sh"
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, f"pure-header gate failed:\n{r.stdout}\n{r.stderr}"
