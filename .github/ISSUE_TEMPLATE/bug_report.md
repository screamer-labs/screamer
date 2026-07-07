---
name: Bug report
about: Report incorrect behavior, a crash, or a wrong result
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear description of what went wrong.

**To reproduce**
A minimal, self-contained snippet that shows the problem:

```python
import numpy as np
import screamer
# ...
```

**Expected behavior**
What you expected to happen (and, if relevant, the reference value from
NumPy/pandas/hand calculation).

**Actual behavior**
What actually happened. Include the full error/traceback if there is one.

**Environment**
- screamer version: (`python -c "import screamer; print(screamer.__version__)"`)
- Python version:
- OS:
- Installed from: wheel (pip) / source

**Additional context**
Anything else that helps, e.g. whether it happens in batch, streaming, or both.
