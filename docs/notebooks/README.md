# Demo notebooks

Runnable notebooks showcasing screamer, using only the public API. Notebooks
01-10 are self-contained with seeded synthetic data. The microstructure
notebooks (11-12) read a small committed real-data slice from `data/` (six hours
of Deribit BTC- and ETH-perpetual trades). Everything runs offline with no
downloads.

## Run / verify

```bash
make notebooks     # builds the lib + bindings, then executes every notebook
# or, against an already-built install:
poetry run pytest --nbmake docs/notebooks/
```

Open any `.ipynb` in Jupyter to read/edit. A notebook that computes an equality
(e.g. batch == stream) asserts it in-cell, so a regression fails the run above.
