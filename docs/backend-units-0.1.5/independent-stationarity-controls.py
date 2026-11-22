"""Independent behavior controls, not source-shaped implementation tests."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import warnings
import numpy as np
from forecastscope.diagnostics import stationarity as st

source = Path(st.__file__)
scales = (1e-9, 1.0, 1e9)
rng = np.random.default_rng(73019)
trend = np.linspace(0.0, 1.0, 120)
walk = np.cumsum(rng.normal(size=240))
seasonal = 3.0 * np.sin(2 * np.pi * np.arange(240) / 12) + rng.normal(0, .05, 240)
results = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, values in (("trend", trend), ("random_walk", walk), ("seasonal", seasonal)):
        rows = []
        for scale in scales:
            for offset_ratio in (0.0, 17.0, -23.0):
                y = scale * (values + offset_ratio)
                rows.append({"scale": scale, "offset_ratio": offset_ratio,
                             "d": st.ndiffs(y), "D": st.nsdiffs(y, 12),
                             "strength": st.seasonal_strength(y, 12)})
        results[name] = rows
        assert len({row["d"] for row in rows}) == 1, (name, "d", rows)
        assert len({row["D"] for row in rows}) == 1, (name, "D", rows)
        strengths = [row["strength"] for row in rows]
        if name != "trend":
            assert max(strengths) - min(strengths) < 1e-10, (name, "strength", rows)
    assert results["trend"][0]["d"] == 1
    assert results["seasonal"][0]["D"] == 1
    constants = []
    for level in (0.0, .1, 1e-14, 1e12, -1e12):
        y = np.full(120, level)
        row = {"level": level, "d": st.ndiffs(y), "D": st.nsdiffs(y, 12),
               "strength": st.seasonal_strength(y, 12)}
        constants.append(row)
        assert (row["d"], row["D"], row["strength"]) == (0, 0, 0.0), row
    reports = []
    report_input = .01 * np.arange(180) + np.sin(2 * np.pi * np.arange(180) / 12) + np.random.default_rng(55).normal(0, .4, 180)
    for scale in scales:
        report = st.stationarity_report(scale * report_input, period=12)
        reports.append({"scale": scale, "d": report["recommended_d"],
                        "D": report["recommended_D"],
                        "strength": report["seasonal_strength"]})
    assert len({(r["d"], r["D"]) for r in reports}) == 1, reports
print(json.dumps({"state": "pass", "distribution_version": md.version("forecastscope"),
                  "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "numpy": np.__version__, "cases": results, "constants": constants, "reports": reports}))
