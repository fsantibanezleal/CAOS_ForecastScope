# Changelog

## 0.01.001 - 2026-11-11

- Return GARCH conditional volatility in source units after internal rescaling.
  Fit results and volatility reports expose the estimation scale for parameters and log-likelihood.
- Derive the Box-Cox positivity shift from the observed series. An all-zero series has no positive scale and raises a clear error.

## 0.01.000 - unreleased

- Extract finite-prefix forecasting, five-expert combinations and sequence
  diagnostics with explicit originating source bindings.
- Add strict public request and forecast-output boundaries.
- Preserve numerical adversarial controls; product adoption remains pending.
