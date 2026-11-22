"""Compare public diagnostics to an explicitly supplied, SHA-verified source archive.

This verification tool executes archived code only as an independent reference.
It never admits archived code into the application or changes retained outcomes.
Run with --archive and its original --manifest, plus an external --out receipt.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType

for _key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')

import numpy as np

EXPECTED_ARCHIVE_SHA256 = '4c53b62e8aff184818adcd07f345be9d4587de99b40d7743aad5554e9ad21413'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def procedures(source):
    return {node.name: digest(ast.dump(node, include_attributes=False).encode())
            for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def equal(actual, expected, path='', atol=0.0):
    if is_dataclass(actual):
        actual = asdict(actual)
    if is_dataclass(expected):
        expected = asdict(expected)
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            equal(actual[key], expected[key], path+'/'+str(key), atol)
    elif isinstance(actual, (list, tuple, np.ndarray)) and isinstance(expected, (list, tuple, np.ndarray)):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            equal(a, b, path+'/'+str(i), atol)
    elif isinstance(actual, (float, np.floating)) or isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=atol, equal_nan=True, err_msg=path)
    else:
        assert actual == expected, (path, actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.manifest.read_text(encoding='utf-8'))
    raw = args.archive.read_bytes()
    assert digest(raw) == receipt['archive_sha256'] == EXPECTED_ARCHIVE_SHA256, 'Immutable reference identity differs'
    originals, audit = {}, {}
    package = ModuleType('forecastscope_archived_diagnostics')
    package.__path__ = []
    sys.modules[package.__name__] = package
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(args.archive) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate archive members'
        assert set(archive.namelist()) == set(receipt['files']), 'Archive inventory differs'
        for name in archive.namelist():
            assert digest(archive.read(name)) == receipt['files'][name], name
        for source in sorted((root/'src/forecastscope/diagnostics').glob('*.py')):
            if source.name == '__init__.py':
                continue
            name = 'data-pipeline/chronoscopelab/analysis/'+source.name
            old = archive.read(name)
            new = source.read_bytes()
            x, y = procedures(old), procedures(new)
            changed = sorted(k for k in x.keys() | y.keys() if x.get(k) != y.get(k))
            audit[source.stem] = {'archive_sha256': digest(old), 'public_sha256': digest(new),
                                  'changed_procedures': changed, 'archive_procedures': x, 'public_procedures': y}
            module = ModuleType(package.__name__+'.'+source.stem)
            module.__file__ = 'verified-archive/'+name
            sys.modules[module.__name__] = module
            # The complete immutable reference identity and all members were verified above.
            exec(compile(old, module.__file__, 'exec'), module.__dict__)  # noqa: S102
            originals[source.stem] = module
    # The frozen product volatility module already delegates to the public package.
    # Its numerical implementation is exactly the originally published 0.1.1 file.
    volatility = root/'src/forecastscope/diagnostics/volatility.py'
    initial = subprocess.check_output(['git', 'show', 'v0.01.001:src/forecastscope/diagnostics/volatility.py'], cwd=root)
    assert initial == volatility.read_bytes(), 'Volatility differs from the original public 0.1.1 implementation'
    audit['volatility']['original_public_0.1.1_sha256'] = digest(initial)
    rng = np.random.default_rng(20261122)
    t = np.arange(320)
    periodic = 20 + .015*t + np.sin(2*np.pi*t/24) + .15*rng.normal(size=len(t))
    walk = 30 + np.cumsum(rng.normal(size=len(t)))
    cause = rng.normal(size=len(t))
    effect = np.roll(cause, 2) + .3*rng.normal(size=len(t))
    logistic = np.empty(480)
    logistic[0] = .4
    for i in range(1, len(logistic)):
        logistic[i] = 3.99*logistic[i-1]*(1-logistic[i-1])
    probes = [
        ('autocorrelation', 'autocorrelation_report', (periodic,), {'nlags': 16, 'lags_m': 10}),
        ('stationarity', 'stationarity_report', (walk,), {'period': 24}),
        ('seasonality', 'seasonality_report', (periodic,), {'candidate_periods': [12, 24]}),
        ('seasonality', 'mstl_decompose', (periodic,), {'periods': [12, 24]}),
        ('causality', 'causality_report', (cause, effect), {'max_lag': 4}),
        ('changepoints', 'pelt', (walk,), {}),
        ('changepoints', 'binseg', (walk,), {'n_bkps': 2}),
        ('changepoints', 'cusum_stability', (periodic,), {}),
        ('distribution', 'distribution_report', (periodic,), {}),
        ('filters', 'hodrick_prescott', (periodic,), {}),
        ('filters', 'christiano_fitzgerald', (periodic,), {}),
        ('filters', 'baxter_king', (periodic,), {}),
        ('filters', 'emd', (periodic,), {'max_imfs': 3}),
        ('fractal', 'fractal_report', (periodic,), {}),
        ('fractal', 'dcca_coefficient', (periodic, walk), {'scale': 24}),
        ('nonlinear', 'nonlinear_report', (logistic,), {'n_surrogates': 3, 'seed': 17}),
        ('sequence', 'matrix_profile', (periodic[:128],), {'window': 16}),
        ('sequence', 'multiscale_entropy', (periodic,), {'scales': [1, 2, 4]}),
        ('volatility', 'volatility_report', (periodic,), {'season_length': 24}),
    ]
    results = []
    for family, name, inputs, keywords in probes:
        original = getattr(originals[family], name)(*inputs, **keywords)
        current = getattr(importlib.import_module('forecastscope.diagnostics.'+family), name)(*inputs, **keywords)
        # Independent normalization only changes dimensionless CCF roundoff.
        equal(current, original, family+'/'+name, atol=5e-14 if family == 'causality' else 0)
        results.append({'family': family, 'procedure': name, 'state': 'pass',
                        'input_sha256': [digest(np.asarray(x, dtype='<f8').tobytes()) for x in inputs],
                        'keywords': keywords})
        print('PASS', family, name, flush=True)
    output = {'state': 'pass', 'archive_sha256': receipt['archive_sha256'],
              'checker_sha256': digest(Path(__file__).read_bytes()),
              'candidate_version': importlib.import_module('forecastscope').__version__,
              'modules_audited': len(audit), 'independent_comparisons': len(results),
              'relative_comparison_tolerance': 5e-12,
              'absolute_tolerance': 'zero except dimensionless CCF roundoff 5e-14',
              'audit': audit, 'comparisons': results,
              'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__,
                          'packages': {name: importlib.metadata.version(name) for name in
                                       ['scipy', 'statsmodels', 'arch', 'nolds', 'stumpy', 'ruptures']}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
