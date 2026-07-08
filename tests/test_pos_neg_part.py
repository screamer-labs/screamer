import numpy as np
from screamer import PosPart, NegPart

def test_pos_neg_part_matches_numpy():
    x = np.array([-2.0, -0.5, 0.0, 1.5, np.nan, 3.0])
    np.testing.assert_array_equal(np.asarray(PosPart()(x)), np.where(np.isnan(x), x, np.maximum(x, 0.0)))
    np.testing.assert_array_equal(np.asarray(NegPart()(x)), np.where(np.isnan(x), x, np.maximum(-x, 0.0)))
    # decomposition identity: x == PosPart(x) - NegPart(x) on finite values
    fin = x[np.isfinite(x)]
    np.testing.assert_allclose(np.asarray(PosPart()(fin)) - np.asarray(NegPart()(fin)), fin)
