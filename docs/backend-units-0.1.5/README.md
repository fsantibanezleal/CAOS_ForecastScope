# Bounded diagnostic boundary corrections for 0.1.5

These controls reproduce diagnostic numerical failures after changing only the
measurement unit or using an exactly constant input. They are generated
regression fixtures, not new forecasting experiments. Historical studies,
published distributions and their source bindings are unchanged.

The separate reader first executed the retained `provider14` receipts against
the actual installed public 0.1.4 wheel. Each receipt records its own runtime,
source identities and observations. The pair/cost/regime receipt includes exact
little-endian float64 input identities. `fixtures.json` binds the author fixture
definitions in `tests/test_diagnostic_boundaries.py`; these reproduce the seeds,
lengths, unit grid and response arrays independently of author output checksums.

| Boundary | Correction | Meaning retained |
| --- | --- | --- |
| Differencing and unit-root tests (C7/C8) | Prefix-derived dimensionless differencing and positive magnitude scaling at five backends; canonical seasonal-strength helper | Explicit no-intercept hypotheses retain levels; exact constants need no differencing |
| EMD (C9) | Fit centered observations divided by their observed maximum deviation | Restore IMF amplitudes and residual levels; test component identity as well as reconstruction |
| ARCH-LM (C10) | Scale the supplied/demeaned residuals before the squared-residual auxiliary regression | Preserve caller's `demean=False` choice and the ARCH test/null |
| Granger/Johansen | Independently scale each observable positively | Preserve lag/null specifications and independent units |
| Normal/RBF change-point costs | Min-max condition these dimensionless costs, including the zero-median RBF fallback | Other costs, including l2 and its squared-unit penalty, remain unchanged |
| Markov mean/variance regimes | Fit min-max-conditioned input in the intercept-bearing model | Return regime means in source units and preserve fitted probability labels |
| Exact-constant normalized diagnostics (C11/C12) | Detect zero observed range on the actual input, including DCCA's shared truncated prefix | Do not manufacture autocorrelation, spectral/chaos/fractal evidence or KDE density from mean-roundoff |
| Hurst failure state (C13) | Isolate backend NumPy error state on success/failure and before fallback | Preserve the caller's process-global arithmetic policy |

Exact-constant raw levels, lag pairs, time-delay embeddings and IAAFT surrogate
values remain meaningful. Distribution reports retain exact mean, zero standard
deviation and a flat QQ line. Undefined normalized shape/normality/complexity
fields are null; KDE/catch22 and matrix-profile availability are explicit.
Normalized inferential entry points that require variation raise `ValueError`.
A nonfinite BDS result supplies no IID decision and emits an availability warning.

The independent bounded sweep also exercised genuine varying observations across
all eleven diagnostic families. Its retained four-family receipt contains 2,056
scalar scale checks with appropriate physical transformations, including inverse
KDE units, QQ slopes, raw moment fields and independently scaled DCCA. These are
finite controls, not a universal numerical-invariance guarantee.

The migration checker authenticates the original 72-file ZIP and every member
before execution. For volatility it additionally protects every original public
0.1.1 procedure AST except the explicitly changed `arch_lm`. The final migration
receipt records both the unconditioned original comparison and the precise
independently conditioned reference recipe for changed boundaries. It never
rewrites the archived source or silently expands the comparison tolerance.
