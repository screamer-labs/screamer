print("TEST PARAM_CASES")
from itertools import product
import numpy as np
from devtools import get_constructor_arguments, get_baselines, sii, get_module_public_classes
import pytest

screamer_module = sii.load_screamer_module()

# ----------------------------------------------------------------------
# Set of classes in the screamer module
# ----------------------------------------------------------------------

# List of all screamer class names
# screamer_classes = [cls for cls in dir(screamer_module) if cls[0].isupper() and cls not in helperClasses]
# The microstructure operators are pure-Python wrappers/compositions, not C++
# functors, so they are outside this harness (which drives the C++ binding surface
# - tensor, view, matrix, stream-vs-batch). Their causality and batch == stream are
# covered by tests/test_microstructure.py; exclude them here.
import screamer.microstructure as _micro
_PYTHON_OPERATORS = set(_micro.__all__)
screamer_classes = [c for c in get_module_public_classes(screamer_module)
                    if c not in _PYTHON_OPERATORS]

# The Rolling classes that fit the standard 1-input/1-output auto-test
# pattern. Exclusions:
#   RollingQuantile  - takes an extra `quantile` arg
#   RollingCorr,
#   RollingCov,
#   RollingBeta,
#   RollingSpread    - all take 2 inputs (FunctorBase<_, 2, 1>); they need
#                      bespoke tests that feed two parallel arrays. See
#                      tests/test_rolling_corr.py and friends.
_ROLLING_AUTO_EXCLUDE = {
    'RollingQuantile',
    # extra non-window arg (mar / threshold / alpha) - tested in test_risk_stats.py
    'RollingDownsideDeviation', 'RollingOmega', 'RollingCVaR',
    # 2-input (FunctorBase<_, 2, 1>) - need parallel arrays
    'RollingCorr', 'RollingCov', 'RollingBeta', 'RollingSpread',
    # 1-input M>1-output (FunctorBase<_, 1, M>) - output shape is
    # (..., M), incompatible with the 1-in/1-out auto harness.
    'RollingMinMax',
    # BollingerBands also has different shape (..., 3) and needs
    # bespoke tests.
    # 2-input range-based volatility (H, L)
    'RollingParkinsonVar', 'RollingParkinsonVol',
    # 4-input OHLC range-based volatility
    'RollingGarmanKlassVar', 'RollingGarmanKlassVol',
    'RollingRogersSatchellVar', 'RollingRogersSatchellVol',
    'RollingYangZhangVar', 'RollingYangZhangVol',
    # 4-input OHLCV volume-aware (FunctorBase<_, 4, 1>)
    'RollingVWAP',
    # 2-input performance metric (FunctorBase<_, 2, 1>)
    'RollingInfoRatio',
    # 2-input regression family (FunctorBase<_, 2, _>)
    'RollingAlpha', 'RollingResidualStd', 'RollingLinearRegression',
    # Multi-scale R/S analysis: needs window_size >= 4 * min_scale = 16,
    # which the default auto-harness window_size=20 only barely satisfies
    # with min_scale=4 and 2 scales. Bespoke tests in test_rolling_hurst.py.
    'RollingHurst',
}
rolling_classes = [cls for cls in screamer_classes
                   if cls.startswith('Rolling') and cls not in _ROLLING_AUTO_EXCLUDE]

# The Ew classes, except:
#   EwSkew, EwKurt: todo baselines
#   EwCov, EwCorr, EwBeta: 2-input (FunctorBase<_, 2, 1>); tested in
#                          tests/test_ew_pair.py
_EW_AUTO_EXCLUDE = {
    # todo baselines
    'EwSkew', 'EwKurt',
    # 2-input pair stats; tested in test_ew_pair.py
    'EwCov', 'EwCorr', 'EwBeta',
    # 2-input range-based volatility (H, L)
    'EwParkinsonVar', 'EwParkinsonVol',
    # 4-input OHLC range-based volatility
    'EwGarmanKlassVar', 'EwGarmanKlassVol',
    'EwRogersSatchellVar', 'EwRogersSatchellVol',
}
ew_classes = [cls for cls in screamer_classes
              if cls.startswith('Ew') and cls not in _EW_AUTO_EXCLUDE]

# Classes that have no arguments. Some no-arg classes are multi-input
# (FunctorBase<_, N, _>) or multi-output (FunctorBase<_, _, M>), which the
# 1-in/1-out auto harness can't drive. Tested in their own files instead.
_NO_ARG_AUTO_EXCLUDE = {
    # 2-input (FunctorBase<_, 2, _>) - need two parallel arrays
    'Hypot', 'Atan2', 'Cart2Polar', 'Polar2Cart',
    # binary arithmetic (FunctorBase<_, 2, 1>) - tested in test_arithmetic.py
    'Add', 'Sub', 'Mul', 'Div',
    # 4-input OHLC (FunctorBase<_, 4, 1>) - tested in test_oscillators_hlc.py
    'BOP',
    # 3-input HLC (FunctorBase<_, 3, 1>) - tested in test_atr_family.py
    'TrueRange',
    # 2-input volume (FunctorBase<_, 2, 1>) - tested in test_adx_and_volume.py
    'OBV',
    # 4-input OHLCV (FunctorBase<_, 4, 1>) - tested in test_adx_and_volume.py
    'AD', 'ADOSC',
    # 2-input comparison / logic masks (FunctorBase<_, 2, 1>) - tested in test_logic_ops.py
    'GreaterThan', 'LessThan', 'GreaterEqual', 'LessEqual',
    'Equal', 'NotEqual', 'And', 'Or',
    # 3-input conditional select (FunctorBase<_, 3, 1>) - tested in test_logic_ops.py
    'Where',
    # 2-input microstructure ops (FunctorBase<_, 2, 1>) - tested in test_microstructure.py
    'OFI', 'LeeReadySign',
    # 4-input micro-price (FunctorBase<_, 4, 1>) - tested in test_microstructure.py
    'MicroPrice',
    # 4-input book-event OFI + 2-input effective spread - tested in test_microstructure.py
    'ContOFI', 'EffectiveSpread',
    # 4-input, 7-output backtest report node - tested in test_backtest.py
    'BacktestReport',
    # 8-input two-sided market-making engine - tested in test_backtest.py
    'BacktestOHLCMaker',
}
# Linear2 takes constructor args (a, b, c) so it is not a no-arg class
# and would not be picked up here - listed for clarity only.
# get_constructor_arguments returns None for a class whose constructor cannot be
# introspected from a pybind11 signature docstring - e.g. the pure-Python
# microstructure operators (OFI, SignedVolume, ...) that have no __init__. Those
# are not C++ functors, so they are outside this parameter sweep; skip them.
no_arg_classes = [
    cls for cls in screamer_classes
    if (ctor_args := get_constructor_arguments(getattr(screamer_module, cls))) is not None
       and len(ctor_args) == 0
       and cls not in _NO_ARG_AUTO_EXCLUDE
]

# ----------------------------------------------------------------------
# Combination test cases (Cartesian products of parameters)
# input arrays default to: "array_length": [100], "array_type": ["default"]
# ----------------------------------------------------------------------

# a list of tuples. A tuple has 2 elements
# 1) a tuple of class_name strings
# 2) a dict param_name: param_values
#  - param_values must be a itterable, e.g. a list
test_definitions = [
    ( tuple(rolling_classes)     , {"window_size": [20]} ),
    ( ('RollingQuantile',)       , {"window_size": [20], "quantile": [0, 0.01, 0.4, 1]} ),
    ( ('RollingPoly1',)          , {"window_size": [20], "derivative_order": [0, 1] }),
    ( ('RollingPoly2',)          , {"window_size": [20], "derivative_order": [0, 1, 2] }),
    ( tuple(ew_classes)          , {"span": [5]}),
    ( tuple(ew_classes)          , {"alpha": [0.2]}),
    ( tuple(ew_classes)          , {"halflife": [10]}),
    ( tuple(ew_classes)          , {"com": [10]}),
    ( tuple(no_arg_classes)      , {"array_type": ["positive"]}),
    ( ('Butter',)                , {"order": [2,3,4,5,6,7,8,9,10], "cutoff_freq": [0.2]}),
    ( ('Diff','Lag')             , {"window_size": [10]}),
    # Despikers (not 'Rolling'-prefixed, so listed explicitly). RollingMedianAD
    # is auto-included via the rolling_classes group above.
    ( ('Hampel','ImpulseClip')   , {"window_size": [20]}),
]


# ----------------------------------------------------------------------
# Define functions for generating special input arrays
# ----------------------------------------------------------------------

def generate_positive_array(array_length):
    """Generate an array of positive values."""
    return np.random.uniform(0.1, 10, array_length)

def generate_array_with_nan(array_length):
    """Generate an array with NaN values interspersed.

    Note: this generator is wired up for use by ``test_stream_vs_batch.py``
    / ``test_stream_vs_generator.py`` but is intentionally NOT referenced
    by the test_definitions table below. The reason is that the cross
    product (function, start_policy, NaN position) of NaN compliance is
    tested explicitly and exhaustively in
    ``tests/test_nan_start_policy_compliance.py`` with stricter assertions
    and per-(function, start_policy) xfail tracking; routing NaN arrays
    through the same generic baseline-parity infrastructure here would
    duplicate that coverage with weaker assertions and would break the
    baseline comparisons (numpy / pandas / TA-Lib references behave
    differently on NaN than screamer's declared policy demands). Keep
    the generator available for ad-hoc tests that need it.
    """
    array = np.random.randn(array_length)
    array[::10] = np.nan  # Insert NaN every 10 elements for testing
    return array

def generate_default_array(array_length):
    """Generate a standard array with random normal values."""
    return np.random.randn(array_length)

def generate_array(array_type, array_length, **kwargs):
    if array_type == 'positive':
        return generate_positive_array(array_length)
    if array_type == 'with_nan':
        return generate_array_with_nan(array_length)
    return generate_default_array(array_length)

# ----------------------------------------------------------------------
# Generate parameter combinations
# ----------------------------------------------------------------------
def generate_combinations(param_dict):
    keys = param_dict.keys()
    values = product(*param_dict.values())
    return [dict(zip(keys, v)) for v in values]


def expanded_test_cases(test_definitions):
    """Generate and return all individual test cases as a dictionary with class groups as keys."""
    all_test_cases = []

    for class_names, param_grid in test_definitions:
        # Generate Cartesian products of parameter combinations
        param_keys = param_grid.keys()
        values = product(*param_grid.values())
        param_instance = [dict(zip(param_keys, combination)) for combination in values]

        # Store the combinations in the dictionary under the class group
        all_test_cases.append((class_names,  param_instance))

    return all_test_cases


# Collect all test cases for pytest parameterization
def yield_test_cases():
    """Yield tuples of (class_name, params_dict, array_type, array_length) for testing."""
    for class_names, param_cases in expanded_test_cases(test_definitions):
        for class_name, params_dict in product(class_names, param_cases):

            # Copy parameters and extract array properties
            params = params_dict.copy()
            array_length = params.pop("array_length", 100)
            array_type = params.pop("array_type", "default")

            yield (class_name, params, array_type, array_length)


def yield_test_cases_with_baselines():
    """Yield tuples of (class_name, baseline_name, params_dict, array_type, array_length) for testing."""
    cases = expanded_test_cases(test_definitions)
    for class_names, param_cases in cases:

        for class_name, params_dict in product(class_names, param_cases):
            # Only yield cases we can actually compare: a class with no reference
            # baseline has nothing to assert against. Baseline coverage gaps are
            # tracked separately by ``python -m devtools.report_baselines``.
            baselines_for_class = get_baselines(class_name)
            if not baselines_for_class:
                continue

            # Copy parameters and extract array properties
            params = params_dict.copy()
            array_length = params.pop("array_length", 100)
            array_type = params.pop("array_type", "default")

            for baseline_name in baselines_for_class:
                yield (class_name, baseline_name, params, array_type, array_length)