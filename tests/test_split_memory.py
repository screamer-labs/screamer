"""split makes at most 2**20 outputs (fix round 1, N2).

split makes one output per source id up to the largest, so split([1.0],
[2**31 - 2]) asked for 2**31 - 1 outputs, which took more memory than a
machine has (the OS killed the process on macOS). The output count is checked
against the cap before anything is allocated: past it, ValueError names the
count and the cap. JavaScript raises the same RangeError (js/test/split.test.ts).
"""
import numpy as np
import pytest

import screamer as sc

CAP = 2**20


def test_split_refuses_more_outputs_than_the_cap():
    with pytest.raises(ValueError, match=r"^split: 2147483647 outputs is more than the cap of 1048576 \(2\*\*20\); pass smaller source ids$"):
        sc.split([1.0], [2**31 - 2])
    with pytest.raises(ValueError, match=r"^split: 1048577 outputs is more than the cap"):
        sc.split([1.0, 2.0], [0, CAP])
    with pytest.raises(ValueError, match=r"^split: n=1048577 outputs is more than the cap"):
        sc.split([1.0], [0], n=CAP + 1)
    # n too small is reported first, as before.
    with pytest.raises(ValueError, match="n=1 is too small for sources up to 2147483646"):
        sc.split([1.0], [2**31 - 2], n=1)


def test_split_at_the_cap_works():
    out = sc.split(np.array([1.0, 2.0]), [0, CAP - 1])
    assert len(out) == CAP
    assert list(out[CAP - 1][0]) == [2.0]
