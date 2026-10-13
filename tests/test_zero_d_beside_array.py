"""A 0-d array beside an array is the mixed-kinds TypeError (final review 9).

A 0-d array is a scalar, so beside an array it gives the TypeError a Python
float beside an array gives, in either argument order: "Add: cannot combine
inputs of types (ndarray, 0-d ndarray); pass all numbers, all arrays, or all
sequences". Two 0-d arrays are two numbers, one event.
"""
import numpy as np
import pytest

import screamer as sc

A = np.array([1.0, 2.0])
Z = np.array(1.0)
MSG = "cannot combine inputs of types ({}); pass all numbers, all arrays, or all sequences"


@pytest.mark.parametrize("op", [sc.Add(), sc.RollingCorr(3), sc.Stoch()],
                         ids=["Add", "RollingCorr", "Stoch"])
def test_a_0d_array_beside_an_array_is_the_mixed_kinds_type_error(op):
    name = type(op).__name__
    k = {"Add": 2, "RollingCorr": 2, "Stoch": 3}[name]
    for pos in range(k):
        args = [A] * k
        args[pos] = Z
        kinds = ", ".join("0-d ndarray" if i == pos else "ndarray" for i in range(k))
        with pytest.raises(TypeError, match=f"^{name}: " + MSG.format(kinds).replace("(", r"\(").replace(")", r"\)")):
            op(*args)
        args[pos] = 1.0                          # a float gives the same form
        kinds = ", ".join("float" if i == pos else "ndarray" for i in range(k))
        with pytest.raises(TypeError, match=MSG.format(kinds).replace("(", r"\(").replace(")", r"\)")):
            op(*args)


def test_0d_arrays_alone_are_one_event():
    assert sc.Add()(Z, Z) == 2.0
    with pytest.raises(TypeError, match=r"\(list, 0-d ndarray\)"):
        sc.Add()([1.0, 2.0], Z)
