"""Independent scale-only backend/model-span controls for provider C8."""
import hashlib, importlib.metadata as md, json, warnings
from pathlib import Path
import numpy as np
from forecastscope.diagnostics import stationarity as st
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron, DFGLS
rng = np.random.default_rng(73019)
rng.normal(size=240)
x = 3 * np.sin(2 * np.pi * np.arange(240) / 12) + rng.normal(0, .05, 240)
rows = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for scale in (1e-9, 1., 1e9):
        y = scale * (x + 17)
        report = st.stationarity_report(y, period=12)
        assert report["recommended_d"] == 0 and report["recommended_D"] == 1
        assert len(report["tests"]) == 5
        for family, option, values in (("adf", "regression", ("n", "c", "ct", "ctt")),
                                       ("kpss", "regression", ("c", "ct")),
                                       ("phillips_perron", "trend", ("n", "c", "ct")),
                                       ("dfgls", "trend", ("c", "ct")),
                                       ("zivot_andrews", "regression", ("c", "t", "ct"))):
            fn = getattr(st, family)
            for value in values:
                actual = fn(y, **{option: value})
                a = y / np.max(np.abs(y))
                if family == "adf":
                    reference_stat = adfuller(a, regression=value, autolag="AIC")[0]
                elif family == "kpss":
                    reference_stat = kpss(a, regression=value, nlags="auto")[0]
                elif family == "phillips_perron":
                    reference_stat = PhillipsPerron(a, trend=value).stat
                elif family == "dfgls":
                    reference_stat = DFGLS(a, trend=value).stat
                else:
                    reference_stat = zivot_andrews(a, regression=value)[0]
                # A centered no-intercept model would fail this comparison.
                np.testing.assert_allclose(actual.stat, reference_stat, rtol=2e-7, atol=2e-8)
                rows.append({"scale": scale, "family": family, option: value,
                             "stat": actual.stat, "pvalue": actual.pvalue,
                             "null": actual.null, "stationary": actual.stationary})
print(json.dumps({"state": "pass", "distribution_version": md.version("forecastscope"),
                  "source": str(st.__file__),
                  "source_sha256": hashlib.sha256(Path(st.__file__).read_bytes()).hexdigest(),
                  "full_report_contexts": 3, "backend_regression_contexts": len(rows), "rows": rows}))
