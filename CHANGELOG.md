# Changelog

## 0.01.005 - 2026-11-22

- Condition KPSS differencing recommendations on dimensionless deviations from the original prefix, rather than treating small measured values as constant via an absolute tolerance. Exact constants need no differences; accumulated floating arithmetic noise after differencing is not tested as a new process.
- Route stationarity's seasonal-strength recommendation through the canonical reviewed seasonal helper, including its exact-constant policy. This removes the divergent duplicate implementation.
- Apply positive observed-magnitude scaling at the five unit-root backend boundaries. Large unit conversions no longer trigger the demonstrated DF-GLS rank failure; input levels and explicit no-intercept hypotheses are preserved.
- Condition EMD stopping, ARCH-LM auxiliary regression, independently scaled Granger/Johansen regressions, normal/RBF change-point costs and intercept-bearing Markov fits on observed dimensionless inputs. EMD components and regime means return in source units. Other change-point costs retain their existing penalty units.
- Refuse normalized inference on exactly constant inputs where it is undefined. Distribution reports retain raw levels and a flat QQ line while returning null shape/complexity verdicts and explicit unavailable KDE/catch22 panels; invalid BDS outputs cannot become an IID decision. Validate DCCA on the shared prefix actually used.
- Restore NumPy's caller error policy even when the Hurst backend changes it before failing.
- Add scale/translation, exact-constant, tiny-process, component-identity and complete-report controls. Forecasting algorithms and retained published studies are unchanged; undefined-input diagnostic fields explicitly use unavailable/null conventions.

## 0.01.004 - 2026-11-22

- Restore the archived nonlinear report contract: the descriptive flag requires Lyapunov strictly above 0.01, zero-one K strictly above 0.5 and the surrogate flag. The numerical decision and seeded RANSAC/OLS algorithms were already unchanged; the previous description misstated its threshold and overstated the conclusion.
- Add strict-boundary, surrogate-condition and isolated-RNG regressions, plus an explicit verified-archive diagnostic migration checker.

## 0.01.003 - 2026-11-22

- Detect exactly constant decomposition inputs and components by their observed range. Floating summation error cannot turn their undefined strength into a unit-dependent value of one.
- Cover STL, MSTL and component strengths at fractional, tiny and large constant levels; numerical forecasting and retained experiment paths are unchanged.

## 0.01.002 - 2026-11-22

- Remove absolute measured-unit floors from contextual features and normalized losses. Constant prefixes supply no normalized-loss training example; combinations use equal experts when no admissible history remains.
- Preserve full precision when constructing forecast result paths.
- Compute decomposition strengths with explicit zero-variance behavior and permit exactly zero remainder variance.
- Keep the default PELT penalty in squared observable units for the l2 cost; other costs require explicit penalties. Exactly constant observations produce one regime.
- Normalize cross-correlation observables independently before products, retaining independent unit invariance across very small and very large scales; reject an undefined constant-series correlation.
- Add independent dimensional controls and retain existing study boundaries; this release claims numerical correctness rather than a new accuracy improvement.

## 0.01.001 - 2026-11-11

- Return GARCH conditional volatility in source units after internal rescaling.
  Fit results and volatility reports expose the estimation scale for parameters and log-likelihood.
- Derive the Box-Cox positivity shift from the observed series. An all-zero series has no positive scale and raises a clear error.

## 0.01.000 - unreleased

- Extract finite-prefix forecasting, five-expert combinations and sequence
  diagnostics with explicit originating source bindings.
- Add strict public request and forecast-output boundaries.
- Preserve numerical adversarial controls; product adoption remains pending.
