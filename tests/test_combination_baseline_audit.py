import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'baseline_report', Path(__file__).resolve().parents[1] / 'scripts/report_combination_baselines.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def rows(blocked=0.):
    return [{'day': day, 'financial_outcomes': {
        'requested_notional': 10000., 'blocked_notional': blocked,
        'unexecuted_notional': 0., 'executed_notional': 10000.-blocked}}
        for day in range(1, 31)]


def score(gaps, policy_rows=None):
    reference = np.full((30, 490), 10000.)
    return report.independent_score(reference, reference-gaps, np.full(490, 10000.),
                                    rows(), policy_rows or rows(), .0038562228090846347)


def test_independent_union_distinguishes_all_three_branches_and_persistence():
    gaps = np.zeros((30, 490))
    assert not score(gaps)['event']
    gaps[0:2, :] = 50.
    assert score(gaps)['first_days'] == {'wealth': 2, 'tail': None, 'function': None}
    gaps[1] = 0.
    gaps[2] = 50.
    assert not score(gaps)['event']
    gaps[:] = 0.
    gaps[5, :5] = 1000.
    assert score(gaps)['first_days'] == {'wealth': None, 'tail': 6, 'function': None}
    gaps[:] = 0.
    assert score(gaps, rows(6000.))['first_days'] == {'wealth': None, 'tail': None, 'function': 2}


def test_net_gain_cannot_trigger_wealth_and_transaction_conservation_is_checked():
    gaps = np.full((30, 490), -200.)
    gaps[:, :200] = 100.
    s = score(gaps)
    assert s['H'][0] > .0038562228090846347 and s['G'][0] < .001
    assert not s['event']
    bad = rows()
    bad[0]['financial_outcomes']['executed_notional'] = 0.
    with pytest.raises(ValueError, match='transaction accounting'):
        score(gaps, bad)


def test_bootstrap_uses_paired_society_differences():
    samples = np.array([[0, 1, 2], [1, 1, 2], [2, 0, 0]])
    a = np.array([1., 10., -5.])
    b = a + .25
    result = report.bootstrap(b-a, samples)
    assert result == {'mean': .25, 'ci95': [.25, .25]}
