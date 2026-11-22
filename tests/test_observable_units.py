"""Independent affine-unit controls for normalization, zero-scale inputs and serialization."""
import numpy as np
import pytest

from forecastscope import combine
from forecastscope.combinations.context_memory import context_features
from forecastscope.diagnostics.causality import cross_correlation
from forecastscope.diagnostics.seasonality import _strengths
from forecastscope.forecasting import mf_from_quantiles


@pytest.mark.parametrize("level", [0.1, 1e-14, 1e12])
def test_exact_constant_decomposition_strengths_do_not_follow_summation_roundoff(level):
    pytest.importorskip("statsmodels")
    from forecastscope.diagnostics.seasonality import mstl_decompose, seasonal_strength, stl_decompose

    values = np.full(120, level)
    zeros = np.zeros_like(values)
    assert _strengths(values, zeros, zeros) == (0, 0)
    assert _strengths(zeros, values, zeros) == (0, 0)
    assert seasonal_strength(values, 12) == 0
    for decomposition in (stl_decompose(values, 12), mstl_decompose(values, [12, 24])):
        assert decomposition.strength_trend == decomposition.strength_seasonal == 0


@pytest.mark.parametrize("factor", [1e-9, 1e9])
def test_context_features_and_allocations_do_not_depend_on_the_measurement_unit(factor):
    y = 3 + .02 * np.arange(64) + np.tile([0., 2., 1., -1.], 16)
    np.testing.assert_allclose(context_features(factor * y, 4), context_features(y, 4), rtol=1e-10, atol=1e-11)
    for method in ("ContextMemory", "ContextMemoryCV", "ValidationMix"):
        original = combine(method, y, seasonality=4, horizon=4)
        converted = combine(method, factor * y, seasonality=4, horizon=4)
        np.testing.assert_allclose(converted.weights, original.weights, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(converted.quantiles / factor, original.quantiles, rtol=1e-8, atol=1e-9)
        assert converted.origins == original.origins


@pytest.mark.parametrize("level", [0., 1e-14, 1e12])
def test_constant_prefix_has_no_invented_loss_scale_and_uses_equal_experts(level):
    for method in ("ContextMemory", "ContextMemoryCV", "ValidationMix"):
        result = combine(method, np.full(64, level), seasonality=4, horizon=4)
        assert result.origins == ()
        np.testing.assert_allclose(result.weights, .2)
        np.testing.assert_allclose(result.quantiles, level, rtol=1e-12, atol=1e-30)


def test_decomposition_strength_distinguishes_exact_components_from_constant_data():
    trend = np.arange(24.)
    seasonal = np.tile([-1., 0., 1., 0.], 6)
    zero = np.zeros(24)
    assert _strengths(trend, seasonal, zero) == (1., 1.)
    assert _strengths(zero, zero, zero) == (0., 0.)
    for factor in (1e-9, 1e9):
        np.testing.assert_allclose(_strengths(factor * trend, factor * seasonal, factor * np.sin(trend)),
                                   _strengths(trend, seasonal, np.sin(trend)), rtol=1e-12)


def test_cross_correlation_refuses_a_constant_observable_and_is_unit_invariant():
    x = np.sin(np.arange(48.) / 3)
    y = np.cos(np.arange(48.) / 4)
    with pytest.raises(ValueError, match="varying series"):
        cross_correlation(x, np.ones(48), max_lag=5)
    expected = cross_correlation(x, y, max_lag=5)
    for xunit, yunit in [(1e-9, 1e-9), (1e-100, 1e100), (1e100, 1e-100)]:
        actual = cross_correlation(xunit * x, yunit * y, max_lag=5)
        np.testing.assert_allclose(actual.values, expected.values, atol=1e-14)
        assert actual.peak_lag == expected.peak_lag


def test_small_measured_predictions_retain_full_precision_at_the_result_boundary():
    columns = np.array([[1.123456789e-11, 2.234567891e-11, 3.345678912e-11]])
    result = mf_from_quantiles("small", "reference", columns, (.1, .5, .9))
    np.testing.assert_array_equal([result.lower, result.point, result.upper], columns.T)


def test_pelt_default_penalty_scales_with_squared_measurement_units():
    pytest.importorskip("ruptures")
    from forecastscope.diagnostics.changepoints import pelt
    x = np.r_[np.zeros(30), np.ones(30)] + .01 * np.sin(np.arange(60))
    expected = pelt(x)
    for factor in (1e-9, 1e9):
        converted = pelt(factor * x)
        assert converted.breakpoints == expected.breakpoints
        np.testing.assert_allclose(converted.penalty / factor ** 2, expected.penalty, rtol=1e-12)


@pytest.mark.parametrize("model", ["rbf", "normal"])
def test_pelt_non_l2_cost_cannot_inherit_a_squared_observable_penalty(model):
    pytest.importorskip("ruptures")
    from forecastscope.diagnostics.changepoints import pelt
    x = np.sin(np.arange(60.))
    with pytest.raises(ValueError, match="explicit penalty"):
        pelt(x, model=model)
    assert pelt(x, model=model, penalty=10.).penalty == 10.


@pytest.mark.parametrize("level", [0., 1e-14, 1e12])
def test_pelt_constant_observations_have_one_regime_in_every_unit(level):
    from forecastscope.diagnostics.changepoints import pelt
    result = pelt(np.full(60, level))
    assert result.breakpoints == []
    assert len(result.segments) == 1
    assert result.segments[0]["start"] == 0 and result.segments[0]["end"] == 60
    assert result.penalty == 0.
