"""Adversarial public boundaries and independent transformation properties."""
import numpy as np
import pytest

from forecastscope import METHOD_NAMES, combine, forecast


@pytest.mark.parametrize("history", [[1., np.nan, 2.], [np.nan] * 4, [1., np.inf, 3.], [[1., 2.], [3., 4.]], [1.]])
def test_nonfinite_or_nonsequence_data_cannot_be_imputed_into_a_success(history):
    with pytest.raises(ValueError, match="contiguous finite"):
        forecast("SES", history)
    with pytest.raises(ValueError, match="contiguous finite"):
        combine("ContextMemory", history)


@pytest.mark.parametrize("setting", [{"horizon": True}, {"horizon": 2.5}, {"horizon": 0}, {"seasonality": 0}, {"seasonality": 4}, {"quantiles": [0, .5, .9]}, {"quantiles": [.9, .5, .1]}, {"quantiles": [.1, .1]}, {"quantiles": [.1, np.nan]}])
def test_invalid_request_does_not_fall_back_to_a_different_method(setting):
    with pytest.raises(ValueError):
        forecast("Holt", [2., 3., 4., 5.], **setting)


def test_validated_combination_refuses_unimplemented_probability_grid():
    with pytest.raises(ValueError, match="require quantiles"):
        combine("ValidationMix", np.arange(50.), quantiles=(.05, .5, .95))


def test_public_forecasts_are_equivariant_under_positive_affine_measurement_changes():
    y = 10 + np.tile([1., 4., 2., 3.], 20) + .03 * np.arange(80)
    for method in METHOD_NAMES:
        original = forecast(method, y, seasonality=4, horizon=9)
        transformed = forecast(method, 17 + 8 * y, seasonality=4, horizon=9)
        np.testing.assert_allclose(transformed, 17 + 8 * original, rtol=1e-10, atol=1e-9)


def test_public_seasonal_prediction_keeps_phase_and_full_requested_horizon():
    y = np.array([10., 20., 30., 10., 20.])
    columns = forecast("SeasonalNaive", y, seasonality=3, horizon=8)
    np.testing.assert_array_equal(columns[:, 1], [30., 10., 20., 30., 10., 20., 30., 10.])
    assert columns.shape == (8, 3)


def test_unknown_names_and_reserved_targets_are_refused():
    with pytest.raises(ValueError, match="Unknown method"):
        forecast("invented", np.arange(20.))
    with pytest.raises(ValueError, match="Unknown combination"):
        combine("invented", np.arange(20.))
    with pytest.raises(TypeError):
        combine("ValidationMix", np.arange(20.), actual=np.arange(3.))
