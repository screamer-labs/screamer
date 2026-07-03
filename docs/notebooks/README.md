# Demo notebooks

Runnable notebooks showcasing screamer. Each is self-contained (seeded synthetic
data, no downloads) and uses only the public API.

## Run / verify

```bash
make notebooks     # builds the lib + bindings, then executes every notebook
# or, against an already-built install:
poetry run pytest --nbmake docs/notebooks/
```

Open any `.ipynb` in Jupyter to read/edit. A notebook that computes an equality
(e.g. batch == stream) asserts it in-cell, so a regression fails the run above.
