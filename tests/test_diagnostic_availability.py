"""A failed optional diagnostic must be visible and cannot become a positive finding."""
import numpy as np
import pytest

from forecastscope.diagnostics import distribution, nonlinear


def unavailable(*args, **kwargs):
    raise RuntimeError("Controlled unavailable diagnostic")


def test_failed_bds_retains_unavailable_verdict_and_warns(monkeypatch):
    monkeypatch.setattr("statsmodels.tsa.stattools.bds", unavailable)
    with pytest.warns(RuntimeWarning, match="BDS diagnostic unavailable"):
        result = distribution.complexity(np.random.default_rng(5).normal(size=80))
    assert result.bds_stat is None and result.bds_pvalue is None and result.iid is None
    assert np.isfinite(result.permutation_entropy)


def test_failed_surrogates_cannot_become_a_positive_chaos_claim(monkeypatch):
    monkeypatch.setattr(nonlinear, "iaaft_surrogate", unavailable)
    y = np.sin(.23 * np.arange(180)) + np.random.default_rng(4).normal(scale=.15, size=180)
    with pytest.warns(RuntimeWarning, match="Surrogate diagnostic unavailable"):
        result = nonlinear.nonlinear_report(y, emb_dim=3, n_surrogates=2)
    assert result["surrogate_test"]["n_surrogates"] == 0
    assert result["surrogate_test"]["nonlinear_detected"] is None
    assert result["likely_chaotic"] is False
