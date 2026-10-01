import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_qwen_decisions import first_crossing,action_timing,rank_fraction
from wolfgym.evaluation.combination_host import read_json


def test_alarm_threshold_is_strict_and_first():
    assert first_crossing([.5,.6,.7,.9],.6)==2
    assert first_crossing([.5,.6],.6) is None


def test_decision_vs_event_close_has_no_lookahead_off_by_one():
    assert action_timing(11,12)=='before_wait_event_close'
    assert action_timing(12,12)=='after_wait_event_observable'
    assert action_timing(29,None)=='no_wait_event'


def test_rank_ties_are_counted_as_joint_max_not_unique_max():
    assert rank_fraction(4,[2,4,4])==1
    assert rank_fraction(2,[2,4,4])==1/3


def test_saved_full_grid_and_denominators():
    p=Path(__file__).resolve().parents[1]/'data/qwen_decision_diagnostic_20260928/diagnostic.json'
    if not p.exists():
        pytest.skip('private saved Qwen diagnostic artifacts are unavailable')
    r=read_json(p)
    assert len(r['rows'])==16 and len(r['frames'])==480
    assert len({x['group_id'] for x in r['rows']})==8
    for c in ('normal','attack'):
        fs=[x for x in r['frames'] if x['condition']==c]
        assert len(fs)==240
        a=r['summary'][c]['feature_groups']
        assert a['eligible_wait']['n']+a['freeze']['n']+sum(not x['eligible'] for x in fs)==240
    assert sum(len(x['interventions']) for x in r['rows'])==24
