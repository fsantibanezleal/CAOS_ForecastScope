"""Cross-unit and exact-constant controls for real diagnostic backend boundaries."""
import json
from dataclasses import asdict

import numpy as np
import pytest


def relationship_fixtures():
    rng = np.random.default_rng(124)
    cause = rng.normal(size=320)
    effect = .9 * np.roll(cause, 2) + rng.normal(0, .7, 320)
    rng = np.random.default_rng(812)
    walk = np.cumsum(rng.normal(size=240))
    linked = 2 * walk + rng.normal(0, .2, 240)
    return cause, effect, walk, linked


def changepoint_fixtures():
    rng = np.random.default_rng(814)
    variance = rng.normal(size=240) * np.r_[np.ones(120), np.full(120, 3.)]
    rng = np.random.default_rng(328)
    regimes = np.repeat([0., 4., 0., 4.], 60) + rng.normal(0, 1.5, 240)
    return variance, regimes


def emd_fixture():
    t = np.arange(256)
    return np.sin(2*np.pi*t/16) + .3*np.sin(2*np.pi*t/51) + .01*t


def arch_fixture():
    rng = np.random.default_rng(981)
    return rng.normal(size=512) * np.tile(np.r_[np.ones(32), np.full(32, 5.)], 8)


def test_emd_components_not_only_reconstruction_preserve_units_and_translation():
    pytest.importorskip("PyEMD")
    from forecastscope.diagnostics.filters import emd

    original = emd_fixture()
    expected = emd(original, max_imfs=6)
    assert expected.imfs.shape[0] == 2
    for unit, offset in [(1e-9, 0), (1e9, 0), (2, 17)]:
        values = unit * (original + offset)
        result = emd(values, max_imfs=6)
        assert result.imfs.shape == expected.imfs.shape
        np.testing.assert_allclose(result.imfs / unit, expected.imfs, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(result.residual / unit - offset, expected.residual, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose((result.imfs.sum(axis=0) + result.residual) / unit,
                                   values / unit, rtol=1e-12, atol=1e-12)


def test_arch_lm_keeps_variance_clustering_evidence_when_units_change():
    pytest.importorskip("statsmodels")
    from forecastscope.diagnostics.volatility import arch_lm

    values = arch_fixture()
    expected = arch_lm(values)
    assert expected.has_arch
    for unit in (1e-9, 1e9):
        result = arch_lm(unit * values)
        assert result.has_arch
        np.testing.assert_allclose([result.lm_stat, result.f_stat, result.lm_pvalue, result.f_pvalue],
                                   [expected.lm_stat, expected.f_stat, expected.lm_pvalue, expected.f_pvalue],
                                   rtol=1e-9, atol=0)


def test_arch_lm_explicit_no_demeaning_preserves_the_supplied_residual_level():
    diagnostic = pytest.importorskip("statsmodels.stats.diagnostic")
    from forecastscope.diagnostics.volatility import arch_lm

    values = 3 + arch_fixture()
    reference = diagnostic.het_arch(values / np.max(np.abs(values)), nlags=12)
    for unit in (1e-9, 1e9):
        result = arch_lm(unit * values, demean=False)
        np.testing.assert_allclose([result.lm_stat, result.lm_pvalue, result.f_stat, result.f_pvalue],
                                   reference, rtol=1e-9, atol=0)


def test_relationship_tests_keep_independent_observable_units():
    pytest.importorskip("statsmodels")
    from forecastscope.diagnostics.causality import cointegration, granger_causality

    cause, effect, walk, linked = relationship_fixtures()
    expected_g = granger_causality(cause, effect, max_lag=5)
    expected_c = cointegration(walk, linked)
    assert expected_g.causes and expected_g.best_lag == 2
    json.dumps(asdict(expected_g), allow_nan=False)
    for cause_unit, effect_unit in [(1e-9, 1e9), (1e9, 1e-9)]:
        result = granger_causality(cause_unit*cause, effect_unit*effect, max_lag=5)
        assert result.causes and result.best_lag == 2
        np.testing.assert_allclose(list(result.pvalues.values()), list(expected_g.pvalues.values()),
                                   rtol=1e-8, atol=0)
        c = cointegration(cause_unit*walk, effect_unit*linked)
        assert c.both_i1 == expected_c.both_i1
        assert c.johansen_rank_at_least_1 == expected_c.johansen_rank_at_least_1
        np.testing.assert_allclose([c.eg_stat, c.johansen_trace_stat],
                                   [expected_c.eg_stat, expected_c.johansen_trace_stat], rtol=1e-7, atol=0)


@pytest.mark.parametrize("model", ["normal", "rbf"])
def test_dimensionless_segmentation_cost_defaults_do_not_depend_on_units(model):
    pytest.importorskip("ruptures")
    from forecastscope.diagnostics.changepoints import binseg, pelt

    variance, _ = changepoint_fixtures()
    values = variance if model == "normal" else np.r_[np.zeros(100), np.ones(20)]
    point = 120 if model == "normal" else 100
    penalty = 15 if model == "normal" else 3
    for unit in (1e-9, 1., 1e9):
        assert pelt(unit*values, model=model, penalty=penalty).breakpoints == [point]
        assert binseg(unit*values, model=model, n_bkps=1).breakpoints == [point]


def test_markov_means_return_in_source_units_and_probabilities_are_unit_invariant():
    pytest.importorskip("statsmodels")
    from forecastscope.diagnostics.changepoints import markov_regimes

    _, values = changepoint_fixtures()
    expected = markov_regimes(values)
    order = np.argsort(expected.regime_means)
    for unit, offset in [(1e-9, 0), (1e9, 0), (2, 17)]:
        actual = markov_regimes(unit*(values+offset))
        current_order = np.argsort(actual.regime_means)
        np.testing.assert_allclose(np.asarray(actual.regime_means)[current_order]/unit-offset,
                                   np.asarray(expected.regime_means)[order], rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(actual.smoothed_probabilities[:, current_order],
                                   expected.smoothed_probabilities[:, order], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("level", [0., .1, 1e-14, 1e12])
def test_exact_constants_have_raw_levels_but_no_fabricated_normalized_inference(level):
    pytest.importorskip("statsmodels")
    pytest.importorskip("antropy")
    pytest.importorskip("nolds")
    pytest.importorskip("stumpy")
    pytest.importorskip("PyEMD")
    from forecastscope.diagnostics import autocorrelation as ac
    from forecastscope.diagnostics import distribution as d
    from forecastscope.diagnostics import fractal as f
    from forecastscope.diagnostics import nonlinear as nl
    from forecastscope.diagnostics import stationarity as st
    from forecastscope.diagnostics.filters import emd
    from forecastscope.diagnostics.sequence import matrix_profile
    from forecastscope.diagnostics.volatility import arch_lm

    values = np.full(256, level)
    assert matrix_profile(values, window=12)["available"] is False
    for procedure in (ac.acf, ac.pacf, ac.ljung_box, ac.box_pierce, d.kde_curve,
                      f.hurst, f.mfdfa, f.fractal_dimension, nl.chaos_measures,
                      nl.zero_one_test, st.adf, st.kpss, arch_lm):
        with pytest.raises(ValueError, match="varying"):
            procedure(values)
    with pytest.raises(ValueError, match="varying"):
        f.dcca_coefficient(values, values, scale=16)
    raw, lagged = ac.lag_plot_pairs(values)
    np.testing.assert_array_equal(raw, values[:-1])
    np.testing.assert_array_equal(lagged, values[1:])
    np.testing.assert_array_equal(nl.time_delay_embed(values), level)
    np.testing.assert_array_equal(nl.iaaft_surrogate(values, n_iter=2), values)
    modes = emd(values)
    assert modes.imfs.shape == (0, len(values))
    np.testing.assert_array_equal(modes.residual, values)
    report = d.distribution_report(values)
    assert report["summary"]["mean"] == level and report["summary"]["std"] == 0
    assert report["summary"]["normal"] is None and report["complexity"]["iid"] is None
    assert report["kde"]["available"] is False and report["catch22"]["available"] is False
    assert report["qq"]["slope"] == 0 and report["qq"]["intercept"] == level
    assert report["qq"]["r"] is None
    json.dumps(report, allow_nan=False)


def test_hurst_backend_failure_cannot_change_numpy_error_policy(monkeypatch):
    backend = pytest.importorskip("hurst")
    pytest.importorskip("nolds")
    from forecastscope.diagnostics.fractal import hurst

    def leaking_backend(*args, **kwargs):
        np.seterr(all="raise")
        raise ValueError("backend failed after changing error state")

    monkeypatch.setattr(backend, "compute_Hc", leaking_backend)
    with np.errstate(all="warn"):
        before = np.geterr()
        result = hurst(np.random.default_rng(91).normal(size=256))
        assert np.isfinite(result.hurst_rs)
        assert np.geterr() == before


def test_dcca_validates_variation_in_the_actual_shared_prefix():
    from forecastscope.diagnostics.fractal import dcca_coefficient

    longer = np.r_[np.full(128, .1), np.arange(128)]
    shorter = np.sin(np.arange(128) / 5)
    for first, second in [(longer, shorter), (shorter, longer)]:
        with pytest.raises(ValueError, match="varying"):
            dcca_coefficient(first, second, scale=16)


def test_nonfinite_bds_verdict_is_unavailable_not_false_iid(monkeypatch):
    stats = pytest.importorskip("statsmodels.tsa.stattools")
    pytest.importorskip("antropy")
    from forecastscope.diagnostics.distribution import complexity

    monkeypatch.setattr(stats, "bds", lambda *args, **kwargs: (np.nan, np.nan))
    with pytest.warns(RuntimeWarning, match="unavailable"):
        result = complexity(np.random.default_rng(40).normal(size=128))
    assert result.bds_stat is result.bds_pvalue is result.iid is None
