"""The nonlinear flag is a bounded descriptive decision, not proof of chaos."""
import numpy as np
import pytest

from forecastscope.diagnostics import nonlinear as nl


@pytest.mark.parametrize(
    ('lyapunov', 'zero_one', 'surrogate_rejects', 'flag'),
    [
        (0.009, 0.9, True, False),
        (0.01, 0.9, True, False),
        (0.011, 0.5, True, False),
        (0.011, 0.501, True, True),
        (0.011, 0.9, False, False),
        (float('nan'), 0.9, True, False),
    ],
)
def test_descriptive_flag_requires_all_three_strict_conditions(monkeypatch, lyapunov, zero_one, surrogate_rejects, flag):
    calls = iter([1.0, 4.0, 5.0, 6.0] if surrogate_rejects else [1.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(nl, 'chaos_measures', lambda *a, **k: nl.ChaosMeasures(
        next(calls), lyapunov, 4, 1, False, 'reference'))
    monkeypatch.setattr(nl, 'zero_one_test', lambda *a, **k: zero_one)
    monkeypatch.setattr(nl, 'recurrence_quantification', lambda *a, **k: nl.RQA(.1, .2, .3, 2, .1, 'reference'))
    monkeypatch.setattr(nl, 'iaaft_surrogate', lambda x, **k: x)
    report = nl.nonlinear_report(np.arange(32), n_surrogates=3, seed=4)
    assert report['likely_chaotic'] is flag
    assert report['verdict_note'] == (
        'Descriptive flag requires Lyapunov > 0.01, 0-1 K > 0.5, and the surrogate flag'
    )


def test_seeded_robust_fit_does_not_depend_on_global_numpy_random_state():
    # One explicit outlier makes the robust fit differ from an ordinary line fit.
    x = np.arange(30, dtype=float)
    y = 2.5*x + 3
    y[8] = 400
    before = np.random.get_state()
    first = nl._seeded_slope((x, y, None), seed=17)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    # An unrelated caller's random draws do not become this method's hidden seed.
    np.random.seed(812)
    try:
        second = nl._seeded_slope((x, y, None), seed=17)
    finally:
        np.random.set_state(before)
    assert first == second == pytest.approx(2.5)
