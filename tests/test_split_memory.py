"""split with a source id too large for memory raises a clear ValueError.

split makes one output per source id up to the largest, so split([1.0],
[2**31 - 2]) asks for 2**31 - 1 outputs. Where an allocation fails, the
message says so ("split: 2147483647 outputs need more memory than is
available; pass smaller source ids"); JavaScript raises the same RangeError
(js/test/split.test.ts). The process runs under an address-space limit, which
Linux enforces.
"""
import subprocess
import sys

import pytest

CODE = r"""
import resource
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
import screamer as sc
try:
    sc.split([1.0], [2**31 - 2])
except ValueError as e:
    print("ValueError:", e)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is enforced on Linux only")
def test_split_with_a_huge_source_raises_a_clear_error():
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, timeout=120)
    assert ("ValueError: split: 2147483647 outputs need more memory than is available; "
            "pass smaller source ids") in r.stdout, r.stdout + r.stderr
