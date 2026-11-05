# Contributing

Use a scoped feature branch, validate the installed package, and merge through
`develop` before promoting to `main`. Preserve unrelated edits and active runs.
Publication is a separate step after source, wheel and applicable diagnostic
checks pass. An unpublished candidate must not be described as a consumed release.

Numerical changes require independent examples, a documented method identity,
and tests for the scientific behavior they change. A retained negative result
cannot be deleted or relabeled by a packaging change. Parameter selection and
evaluation outcomes must use separate declared information sets.

Keep NumPy-only forecasting imports independent of optional diagnostic packages.
Application cases, runtime/checkpoint policies, artifact schemas and deployment
workflows belong to their consuming applications. Never add credentials, local
environment paths or source data acquired under a restricted licence.

Run `ruff check src tests`, `python -m pytest tests`, `python -m build` and
`python -m twine check dist/*` in the declared external environment. Diagnostic
tests require the `diagnostics` extra; running only the core tests is not a claim
that the full optional toolkit was validated.
