# Diagnostic migration verification

Version 0.1.4 restores the nonlinear report's bounded description of its existing
decision. The flag already required Lyapunov strictly above 0.01, zero-one K
strictly above 0.5 and the descriptive surrogate flag. Its previous note
incorrectly said Lyapunov above zero and overstated the conclusion as a chaos
claim. The numerical helpers, explicit RANSAC seeds and OLS fallback did not
change.

The verification compares every function and class AST in all eleven diagnostic
modules with the original product archive. It also executes nineteen original
versus public-provider controls on explicit periodic, random-walk, lagged and
logistic-map inputs. Complete source, procedure and input SHA256 values are in
[the machine receipt](diagnostic-migration-0.1.4.json).

| Family | Source differences and their interpretation |
| --- | --- |
| Autocorrelation | Function and class bodies unchanged |
| Stationarity | Function and class bodies unchanged |
| Filters | Equivalent `int(len(...))` simplification in report metadata |
| Fractal | Import order and equivalent `range(0, end)` and integer simplifications |
| Sequence | Equivalent removal of `int` around integer-valued `math.ceil` |
| Distribution | Explicit unavailable-diagnostic warning and equivalent length conversion |
| Nonlinear | Restored bounded description and named thresholds; explicit unavailable warning; numerical helper bodies unchanged |
| Causality | Reviewed independent observable normalization and explicit undefined constant correlation; equivalent length conversion |
| Changepoints | Reviewed squared-unit l2 penalty, caller-supplied non-l2 penalty and constant-input policy; equivalent adjacent-pair iteration |
| Seasonality | Reviewed dimensional strength and exact-constant policies; equivalent integer simplifications |
| Volatility | Original product module already delegates publicly; current numerical implementation is byte-identical to published 0.1.1 |

All nineteen controls passed. Comparisons use relative tolerance 5e-12 and zero
absolute tolerance, except 5e-14 absolute roundoff tolerance for the dimensionless
cross-correlation calculation after its reviewed normalization. The earlier unit
regressions separately test the intentional corrections at tiny, large and
constant observable scales. Equality on these controls does not establish every
diagnostic's statistical validity or future numerical behavior on arbitrary data.

The reference is ChronoScope's complete immutable source archive with SHA256
`4c53b62e8aff184818adcd07f345be9d4587de99b40d7743aad5554e9ad21413`.
The checker requires that exact identity, the original manifest inventory and
every member hash before executing any archived reference. Reference execution
is confined to verification; archived code is never the application's active
implementation. Existing scientific outcomes are not regenerated or relabelled.

To reproduce, supply the explicit local archive and its original companion
manifest from the ChronoScope repository:

```text
python tools/check_diagnostic_migration.py --archive <provider-migration-source.zip> --manifest <provider-migration-source-archive.json> --out <diagnostic-receipt.json>
```

The checker requires the diagnostic extras and the original published 0.1.1 Git
tag for the independent volatility source comparison. Its output identifies the
actual runtime and checker bytes. The library's complete test suite and the
application's installed-provider checks remain separate gates.
