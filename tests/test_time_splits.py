"""time_splits: scikit-learn style time-series folds with label purging."""
import numpy as np
import pytest

import screamer as s


def _folds(folds):
    return [(list(tr), list(te)) for tr, te in folds]


def test_returns_int_array_pairs():
    folds = s.time_splits(12, 3)
    assert isinstance(folds, list) and len(folds) == 3
    for tr, te in folds:
        assert isinstance(tr, np.ndarray) and tr.dtype == np.int64 and tr.ndim == 1
        assert isinstance(te, np.ndarray) and te.dtype == np.int64 and te.ndim == 1


def test_default_folds():
    # test_size = 12 // 4 = 3; test blocks end at n.
    assert _folds(s.time_splits(12, 3)) == [
        (list(range(0, 3)), [3, 4, 5]),
        (list(range(0, 6)), [6, 7, 8]),
        (list(range(0, 9)), [9, 10, 11]),
    ]


@pytest.mark.parametrize("n, n_splits, test_size, max_train_size, gap", [
    (10, 2, None, None, 0), (100, 5, None, None, 0), (101, 5, None, None, 0),
    (50, 3, 7, None, 0), (50, 4, None, 6, 0), (37, 6, 2, 10, 0),
    (6, 5, None, None, 0), (100, 5, None, None, 3), (60, 4, 5, 8, 2),
    (100, 5, None, 15, 7),
])
def test_matches_sklearn(n, n_splits, test_size, max_train_size, gap):
    # An integer label_length is scikit-learn's gap.
    sk = pytest.importorskip("sklearn.model_selection")
    want = list(sk.TimeSeriesSplit(n_splits=n_splits, test_size=test_size,
                                   max_train_size=max_train_size, gap=gap).split(np.zeros(n)))
    got = s.time_splits(n, n_splits, test_size=test_size, label_length=gap,
                        max_train_size=max_train_size)
    assert len(got) == len(want)
    for (gtr, gte), (wtr, wte) in zip(got, want):
        np.testing.assert_array_equal(gtr, wtr)
        np.testing.assert_array_equal(gte, wte)


def test_integer_label_length_purges_the_last_rows():
    folds = s.time_splits(20, 2, test_size=5, label_length=3)
    # Test blocks [10, 15) and [15, 20); a row t trains when t + 3 < start.
    assert _folds(folds) == [(list(range(0, 7)), list(range(10, 15))),
                             (list(range(0, 12)), list(range(15, 20)))]


def test_array_label_length_purges_each_row():
    n = 20
    label = np.zeros(n)
    label[8] = 2      # 8 + 2 = 10 reaches the first test row: dropped from fold 0
    label[7] = 2      # 7 + 2 = 9: kept
    label[3] = 12     # 3 + 12 = 15 reaches both test blocks: dropped from both
    label[13] = 1     # 13 + 1 = 14: kept in fold 1
    label[14] = 1     # 14 + 1 = 15: dropped from fold 1
    folds = s.time_splits(n, 2, test_size=5, label_length=label)
    tr0, te0 = folds[0]
    tr1, te1 = folds[1]
    assert list(te0) == list(range(10, 15)) and list(te1) == list(range(15, 20))
    assert list(tr0) == [0, 1, 2, 4, 5, 6, 7, 9]
    assert list(tr1) == [0, 1, 2] + list(range(4, 14))


def test_array_label_length_equals_integer_when_constant():
    a = s.time_splits(50, 4, label_length=3)
    b = s.time_splits(50, 4, label_length=np.full(50, 3.0))
    for (atr, ate), (btr, bte) in zip(a, b):
        np.testing.assert_array_equal(atr, btr)
        np.testing.assert_array_equal(ate, bte)


def test_rule_holds_for_random_labels():
    rng = np.random.default_rng(3)
    n = 200
    label = rng.integers(0, 15, n).astype(float)
    for tr, te in s.time_splits(n, 4, label_length=label, max_train_size=60):
        start = te[0]
        assert np.all(tr < start)
        assert np.all(tr + label[tr] < start)
        kept = [t for t in range(start) if t + label[t] < start]
        np.testing.assert_array_equal(tr, kept[-60:])


def test_max_train_size_keeps_the_most_recent_kept_rows():
    label = np.zeros(30)
    label[19] = 5     # dropped from the fold that tests [20, 25)
    folds = s.time_splits(30, 2, test_size=5, label_length=label, max_train_size=4)
    assert list(folds[0][0]) == [15, 16, 17, 18]
    assert list(folds[1][0]) == [21, 22, 23, 24]


def test_label_length_inputs():
    # A list works as an array; a fractional length compares as a number.
    folds = s.time_splits(10, 2, test_size=3, label_length=[1.5] * 10)
    assert list(folds[0][0]) == [0, 1, 2]


def test_empty_training_set_raises():
    with pytest.raises(ValueError, match="^time_splits: .*no training rows"):
        s.time_splits(20, 2, test_size=5, label_length=10)
    label = np.zeros(20)
    label[:10] = 50
    with pytest.raises(ValueError, match="^time_splits: .*no training rows"):
        s.time_splits(20, 2, test_size=5, label_length=label)


def test_nan_label_length_raises():
    label = np.zeros(20)
    label[17] = np.nan
    with pytest.raises(ValueError, match=r"^time_splits: label_length\[17\] is NaN; drop the rows "
                       r"without a label first, for example with "
                       r"forecast_pairs\(\.\.\., dropna=True\)$"):
        s.time_splits(20, 2, label_length=label)
    with pytest.raises(ValueError, match="^time_splits: label_length is NaN"):
        s.time_splits(20, 2, label_length=float("nan"))


def test_negative_label_length_raises():
    label = np.zeros(20)
    label[4] = -1
    with pytest.raises(ValueError, match=r"^time_splits: label_length\[4\] must not be negative"):
        s.time_splits(20, 2, label_length=label)
    with pytest.raises(ValueError, match="^time_splits: label_length must not be negative"):
        s.time_splits(20, 2, label_length=-1)


def test_label_length_must_have_length_n():
    with pytest.raises(ValueError, match="^time_splits: label_length has 19 values, expected n = 20"):
        s.time_splits(20, 2, label_length=np.zeros(19))
    with pytest.raises(ValueError, match="^time_splits: label_length must be a number or a 1-D array"):
        s.time_splits(20, 2, label_length=np.zeros((20, 1)))


def test_too_many_splits_raises():
    with pytest.raises(ValueError, match="^time_splits: n_splits = 5 needs at least 6 rows, got n = 5"):
        s.time_splits(5, 5)
    with pytest.raises(ValueError, match="^time_splits: n_splits = 4 test blocks of 5 rows need more than n = 20 rows"):
        s.time_splits(20, 4, test_size=5)


@pytest.mark.parametrize("kw, msg", [
    (dict(n=10, n_splits=1), "n_splits must be an integer of at least 2"),
    (dict(n=10, n_splits=2.5), "n_splits must be an integer of at least 2"),
    (dict(n=-1), "n must be an integer of at least 1"),
    (dict(n=10.5), "n must be an integer of at least 1"),
    (dict(n=10, n_splits=2, test_size=0), "test_size must be an integer of at least 1"),
    (dict(n=10, n_splits=2, max_train_size=0), "max_train_size must be an integer of at least 1"),
])
def test_argument_checks(kw, msg):
    with pytest.raises(ValueError, match=f"^time_splits: {msg}"):
        s.time_splits(**kw)


def test_bool_arguments_raise():
    with pytest.raises(TypeError, match="^time_splits: n must be an integer, got bool$"):
        s.time_splits(True)
    with pytest.raises(TypeError, match="^time_splits: label_length must be a number, got bool$"):
        s.time_splits(10, 2, label_length=True)


def test_usable_as_sklearn_cv():
    sk = pytest.importorskip("sklearn.model_selection")
    lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 2))
    y = X @ [1.0, -1.0] + 0.1 * rng.standard_normal(80)
    scores = sk.cross_val_score(lm.LinearRegression(), X, y,
                                cv=s.time_splits(len(X), 4, label_length=2))
    assert scores.shape == (4,)
