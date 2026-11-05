"""Compare the installed extraction with its exact source and admitted histories."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

import forecastscope
from forecastscope import combine, forecast

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--product", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    product = args.product.resolve()
    origin = json.loads((ROOT / "docs/extraction-source.json").read_text(encoding="utf-8"))
    for name in ("forecasting.py", "combinations/context_memory.py", "combinations/validated.py"):
        entry = origin["files"][name]
        assert hashlib.sha256((product / entry["source_path"]).read_bytes()).hexdigest() == entry["source_sha256"], "Origin source changed"
    sys.path.insert(0, str(product / "data-pipeline"))
    from chronoscopelab.model import forecasters as old
    from chronoscopelab.model.context_memory import fit_context_memory
    from chronoscopelab.model.validated_combination import fit_validated

    artifact = product / "data/derived/fresh-reevaluation.json"
    rows = []
    for case in json.loads(artifact.read_text(encoding="utf-8"))["cases"]:
        y, m, h = np.asarray(case["history"]), case["seasonality"], case["horizon"]
        count = 0
        for name, _, fn in old.METHODS:
            expected = old.gaussian_quantiles(*fn(y, m, h), (.1, .5, .9))
            np.testing.assert_array_equal(forecast(name, y, seasonality=m, horizon=h), expected)
            count += 1
        for name, mode in [("ContextMemory", "context"), ("RecencyMean", "recency"), ("EqualMean", "uniform"), ("ContextMemoryCV", None), ("ValidationMix", None)]:
            expected = fit_context_memory(y, m, h, mode=mode) if mode else fit_validated(y, m, h, stacking=name == "ValidationMix")
            got = combine(name, y, seasonality=m, horizon=h)
            np.testing.assert_array_equal(got.quantiles, expected.quantiles)
            np.testing.assert_array_equal(got.weights, expected.weights)
            assert got.origins == expected.origins
            count += 1
        rows.append({"case_id": case["case_id"], "history_sha256": hashlib.sha256(y.astype("<f8").tobytes()).hexdigest(), "history_length": len(y), "horizon": h, "seasonality": m, "exact_method_comparisons": count})
        print(case["case_id"], count, "exact comparisons", flush=True)
    installed = Path(forecastscope.__file__).resolve() == Path(distribution("forecastscope").locate_file("forecastscope/__init__.py")).resolve()
    result = {"state": "installed-wheel-parity; product adoption separate" if installed else "local-source-parity; installed wheel and product adoption separate", "source_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "cases": rows, "total_exact_comparisons": sum(r["exact_method_comparisons"] for r in rows), "scope": "Full-precision quantile matrices, allocation weights and historical origins; no new accuracy trial", "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    dest = ROOT / "docs/product-parity.json"
    if args.check:
        old_receipt = json.loads(dest.read_text(encoding="utf-8"))
        for key in ("source_artifact_sha256", "cases", "total_exact_comparisons"):
            assert result[key] == old_receipt[key], f"Parity receipt changed: {key}"
    else:
        dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
