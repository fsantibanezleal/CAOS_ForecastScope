# forecastscope

Finite-prefix forecasts, temporal combinations of reference experts, and sequence
diagnostics in a reusable Python library. The core requires NumPy. Diagnostic
implementations use optional scientific packages through `forecastscope.diagnostics`.

Release availability is recorded on [PyPI](https://pypi.org/project/forecastscope/)
and [GitHub](https://github.com/fsantibanezleal/CAOS_ForecastScope/releases).
Application adoption is a separate step. The extraction preserves numerical
equations while adding strict public request and output validation, lint
normalization and warnings for unavailable optional diagnostics. It makes no new forecast
accuracy, calibration, chaos-identification or operational-performance claim.

```python
import numpy as np
from forecastscope import combine, forecast

history = 20 + np.tile([1., 4., 2., 3.], 30)
columns = forecast("HoltWinters", history, seasonality=4, horizon=8)
allocation = combine("ValidationMix", history, seasonality=4, horizon=8)
print(columns.shape, allocation.weights, allocation.origins)
```

The caller supplies a contiguous observed history; raw arrays contain no timestamps,
so contiguity itself is a caller precondition. Future targets belong to the caller's evaluation
protocol. NaNs, infinite values, fractional horizons, invalid seasonal lags and
unsupported probability grids are refused by the public entry points. The reference kernels preserve their documented short-history policies, including
Holt fallback when additive seasonal smoothing lacks two cycles or has lag one.
Combination experts can consequently coincide on a nonseasonal input. The five
reference experts use approximate Gaussian residual intervals; ordered columns do
not establish calibration. Temporal validation filters an outer-prefix-defined
historical grid, and does not reconstruct the exact grid a historical service
would have proposed at every earlier date.

The fixed contextual combination has an unfavorable comparison with its matching
recency control in its original development study. Later validation-selected
combinations have bounded point-score improvements on specified external collections,
with stronger automatic-method and interval-score comparisons retained. The package
does not turn those studies into a general superiority claim.

For diagnostics, install the `diagnostics` extra and import the topic module, for
example `forecastscope.diagnostics.sequence.matrix_profile`. Diagnostics describe
their supplied sequence and can be retrospective. They must not be computed on a
future block and then treated as prefix-only model-selection inputs.

The source origin and exact input hashes are recorded in
[docs/extraction-source.json](docs/extraction-source.json). The originating software
is MIT licensed; its copyright and licence remain in this repository. Application
cases, generator seeds, source CSVs, artifact schemas, pipeline stages and web assets
are outside this library. See [the extraction boundary](docs/boundary.md).

Numerical unit handling in 0.1.2 keeps dimensional scales derived from the
observations. A constant prefix has no nonzero loss scale and therefore supplies
no contextual or temporal-validation training example. With no eligible examples,
the forecast combination uses equal expert weights. This is a declared fallback,
not evidence that the combination learned from a constant history. Exactly
constant observations have one PELT regime; non-l2 PELT costs require a caller-supplied
penalty in the cost's units. Cross-correlation requires both series to vary and
normalizes their units independently before multiplication. Forecast result
objects retain the computed floating-point paths without decimal rounding.

Version 0.1.3 also assigns zero seasonal and trend strengths to exactly constant
observations, including decimal levels for which floating-point variance may be
nonzero. The same policy applies to public seasonal strength, STL and MSTL results.
See [the verified publication receipt](docs/publication-0.1.3.json).

The nonlinear diagnostic report uses a descriptive threshold flag, not proof of
chaos. Version 0.1.4 restores its exact threshold description without changing the
seeded numerical calculations. See the [eleven-family diagnostic migration audit](docs/diagnostic-migration-0.1.4.md)
and [verified public release](docs/publication-0.1.4.json).
