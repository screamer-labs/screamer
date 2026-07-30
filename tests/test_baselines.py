print("TEST: test_baselines.py")
import numpy as np
from .param_cases import SHAPES, generate_arrays, yield_test_cases_with_baselines
from devtools import baselines, sii
import pytest

screamer_module = sii.load_screamer_module()

# Create a pytest parameterization using the collected test cases
@pytest.mark.parametrize(
    "class_name, baseline_name, params, array_type, array_length",
    yield_test_cases_with_baselines()
)
def test_screamer_vs_baseline(class_name, baseline_name, params, array_type, array_length):
    """Compare the output of screamer class and baseline reference implementation."""
    # Get screamer and baseline classes
    screamer_class = getattr(screamer_module, class_name, None)
    baseline_class = getattr(baselines, baseline_name, None)

    if not screamer_class or not baseline_class:
        pytest.skip(f"Skipping {class_name} vs {baseline_name} due to missing implementation")

    # Instantiate the screamer and baseline objects
    screamer_instance = screamer_class(**params)
    baseline_instance = baseline_class(**params)

    # One array per input. A 2-input operator gets two independent draws, so
    # Sub is not compared on all zeros and Equal not on all ones.
    n_inputs, _ = SHAPES.get(class_name, (1, 1))
    input_arrays = generate_arrays(array_type, array_length, n_inputs)

    # Run both implementations
    screamer_output = screamer_instance(*input_arrays)
    baseline_output = baseline_instance(*input_arrays)

    screamer_output = np.asarray(screamer_output)
    baseline_output = np.asarray(baseline_output)
    assert screamer_output.shape == baseline_output.shape, (
        f"{class_name}: baseline returns {baseline_output.shape}, "
        f"operator returns {screamer_output.shape}"
    )

    # compare outputs
    # todo: for now we only compare the last 10 elements because we have no clear
    # policy about what to do with the first window_size - 1 elements.
    # sometimes those are zero, other times NaN
    np.testing.assert_allclose(
        screamer_output[-10:], baseline_output[-10:], rtol=1e-5, atol=1e-8,
        err_msg=f"Results do not match for {class_name} vs {baseline_name} with params {params} and array type '{array_type}' of length {array_length}"
    )

