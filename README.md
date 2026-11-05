# forecastscope

Finite-prefix forecasts, temporal combinations of reference experts, and sequence
diagnostics in a reusable Python library. The core requires NumPy. Diagnostic
implementations use optional scientific packages through `forecastscope.diagnostics`.

This is an extraction candidate. It is not yet a published PyPI dependency, and
ChronoScope has not yet migrated to it. The extraction preserves numerical kernels
while adding strict public request and output validation. It makes no new forecast
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
