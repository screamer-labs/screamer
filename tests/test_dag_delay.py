import numpy as np
from screamer import Delay
from screamer.dag import Input, Pipeline


def test_delay_as_pipeline_node():
    x = Input("x")
    p = Pipeline([x], [Delay(5)(x)])
    v, i = p((np.array([1.0, 2.0, 3.0]), np.array([0, 7, 14], dtype=np.int64)))
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(i, [5, 12, 19])
