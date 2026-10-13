"""G33 (a): the portfolio engines take one packed (T, K, F) array, F last in
the order of the separate arguments, with the results of the separate arrays
bit for bit (rule 4)."""
import numpy as np
import pytest

import screamer as s

RNG = np.random.default_rng(33)
T, K = 120, 6
PX = 100 * np.exp(np.cumsum(RNG.standard_normal((T, K)) * 0.01, axis=0))
OPEN = PX * (1 + RNG.standard_normal((T, K)) * 0.002)
HIGH = np.maximum(OPEN, PX) * 1.003
LOW = np.minimum(OPEN, PX) * 0.997
TGT = np.where(RNG.random((T, K)) < 0.3, np.nan, RNG.standard_normal((T, K)) * 0.2)
PX_NAN = PX.copy()
PX_NAN[RNG.random((T, K)) < 0.05] = np.nan


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


TARGET_OPS = [s.BacktestPortfolioTarget(fee=[0.001], size="weight"),
              s.BacktestPortfolioTarget(fee=[0.0005] * K, size="notional", initial_capital=1e5,
                                        rebalance="all", max_gross=1.5),
              s.BacktestPortfolioTarget(slippage=[0.001], rebalance_band=0.05)]
OHLC_OPS = [s.BacktestPortfolioOHLCTarget(fee=[0.001]),
            s.BacktestPortfolioOHLCTarget(fill_at="close", delay=0, max_net=0.5),
            s.BacktestPortfolioOHLCTarget(fill_at="ohlc4", delay=2, size="units")]


@pytest.mark.parametrize("op", TARGET_OPS)
@pytest.mark.parametrize("price", [PX, PX_NAN], ids=["clean", "nan"])
def test_target_engine_packed_equals_separate(op, price):
    ref = op(TGT, price)
    packed = np.stack([TGT, price], axis=-1)
    assert same(op(packed), ref)
    assert same(op(np.asfortranarray(packed)), ref)
    assert same(op(np.stack([TGT, price, price], axis=-1)[:, :, :2]), ref)      # strided view


@pytest.mark.parametrize("op", OHLC_OPS)
def test_ohlc_engine_packed_equals_separate(op):
    fields = (TGT, OPEN, HIGH, LOW, PX_NAN)
    ref = op(*fields)
    packed = np.stack(fields, axis=-1)
    assert same(op(packed), ref)
    assert same(op(packed.astype(np.float32)), op(*[f.astype(np.float32) for f in fields]))


def test_misaligned_packed_reads_like_aligned():
    op = s.BacktestPortfolioTarget(fee=[0.001])
    packed = np.stack([TGT, PX], axis=-1)
    buf = np.zeros(packed.size * 8 + 1, dtype=np.uint8)
    view = np.frombuffer(buf, dtype=np.float64, count=packed.size, offset=1).reshape(packed.shape)
    buf[1:] = np.frombuffer(packed.tobytes(), dtype=np.uint8)
    assert same(op(view), op(TGT, PX))


def test_empty_and_one_step():
    op = s.BacktestPortfolioTarget()
    assert op(np.empty((0, K, 2))).shape == (0, K, 4)
    assert same(op(np.stack([TGT[:1], PX[:1]], axis=-1)), op(TGT[:1], PX[:1]))


def test_a_packed_array_of_other_fields_raises():
    with pytest.raises(ValueError, match=r"^a packed input of BacktestPortfolioTarget has shape \(T, K, 2\), got \(3, 2, 3\)\.$"):
        s.BacktestPortfolioTarget()(np.ones((3, 2, 3)))
    with pytest.raises(ValueError, match=r"^BacktestPortfolioOHLCTarget expects 5 inputs, or one packed \(T, K, 5\) array, got 2$"):
        s.BacktestPortfolioOHLCTarget()(TGT, PX)
