"""Validated entry points; legacy numerical kernels retain their original equations."""
from __future__ import annotations

import numpy as np

from .combinations.context_memory import MemoryFit, fit_context_memory
from .combinations.validated import ValidatedFit, fit_validated
from .forecasting import METHOD_FNS, gaussian_quantiles

METHOD_NAMES = tuple(METHOD_FNS)


def _request(history, seasonality, horizon, quantiles):
    y = np.asarray(history, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("A contiguous finite one-dimensional history of at least two observations is required")
    for name, value in [("seasonality", seasonality), ("horizon", horizon)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if seasonality >= len(y):
        raise ValueError("seasonality must be smaller than the observed history length")
    levels = tuple(float(q) for q in quantiles)
    if not levels or not np.isfinite(levels).all() or not all(0 < q < 1 for q in levels) or not all(a < b for a, b in zip(levels, levels[1:])):
        raise ValueError("Quantiles must be finite, strictly increasing probabilities between zero and one")
    return y, int(seasonality), int(horizon), levels


def _output(values, horizon, levels):
    a = np.asarray(values, dtype=float)
    if a.shape != (horizon, len(levels)) or not np.isfinite(a).all() or np.any(np.diff(a, axis=1) < 0):
        raise ValueError("The method returned invalid forecast dimensions, nonfinite values or crossing columns")
    return a


def forecast(method: str, history, *, seasonality: int = 1, horizon: int = 1, quantiles=(.1, .5, .9)) -> np.ndarray:
    """Return full-precision horizon-by-quantile Gaussian residual columns.

    Inputs must be contiguous and finite; no imputation, source acquisition or
    seasonal-period selection is performed. These approximate intervals carry no
    calibration guarantee. Methods refit on the whole supplied prefix.
    """
    y, m, h, levels = _request(history, seasonality, horizon, quantiles)
    if method not in METHOD_FNS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHOD_NAMES}")
    point, sigma = METHOD_FNS[method](y, m, h)
    return _output(gaussian_quantiles(point, sigma, levels), h, levels)


def combine(method: str, history, *, seasonality: int = 1, horizon: int = 1, quantiles=(.1, .5, .9)) -> MemoryFit | ValidatedFit:
    """Fit a five-expert combination without accepting a reserved future block.

    The returned object retains weights and complete historical origins. The
    validated variants additionally retain candidate scores and inner splits.
    Their canonical probability grid is fixed; unsupported grids are refused.
    """
    y, m, h, levels = _request(history, seasonality, horizon, quantiles)
    modes = {"ContextMemory": "context", "RecencyMean": "recency", "EqualMean": "uniform"}
    if method in modes:
        result = fit_context_memory(y, m, h, levels, mode=modes[method])
    elif method in ("ContextMemoryCV", "ValidationMix"):
        if levels != (.1, .5, .9):
            raise ValueError("Temporally validated combinations require quantiles (0.1, 0.5, 0.9)")
        result = fit_validated(y, m, h, stacking=method == "ValidationMix")
    else:
        raise ValueError(f"Unknown combination {method!r}")
    _output(result.quantiles, h, levels)
    weights = np.asarray(result.weights)
    if weights.shape != (5,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid expert allocation")
    return result
