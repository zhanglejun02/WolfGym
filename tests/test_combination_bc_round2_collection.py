"""Teacher-label history must follow the student, not teacher suggestions."""
import copy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import collect_combination_bc_round2 as collector
from wolfgym.evaluation.combination_host import read_json


@pytest.fixture
def artifact():
    directory=ROOT/'data/combination_bc_round1_20260927/monitor/bc_unweighted'
    paths=list(directory.glob('*_attack/artifact.json.gz'))
    if not paths:pytest.skip('saved student integration fixture unavailable')
    return read_json(sorted(paths)[0])


def test_labels_keep_student_history_and_private_metadata_separate(artifact,monkeypatch):
    class AggressiveTeacher:
        def __init__(self,*args):pass
        def __call__(self,view):return 1 if view['observations'][-1]['remaining_budget']>=4 else 0
    monkeypatch.setattr(collector,'BaselinePolicy',AggressiveTeacher)
    samples,provenance=collector.label_student_history(artifact)
    assert len(samples)==len(provenance)==30
    assert all(set(s)=={'sample_id','view','action'} for s in samples)
    for day,s in enumerate(samples):
        expected=[{'day':r['day'],'target_id':r['frame']['action_targets'][r['action']-1],
                   'action':'FREEZE','cost':4.} for r in artifact['decisions'][:day] if r['action']]
        assert s['view']['interventions']==expected
        assert provenance[day]['executed_student_action']==artifact['decisions'][day]['action']
    assert any(p['teacher_action']!=p['executed_student_action'] for p in provenance)


def test_changed_student_input_rejected(artifact):
    artifact['decisions'][5]['input_sha256']='bad'
    with pytest.raises(ValueError,match='student input changed'):collector.label_student_history(artifact)


def test_nontrain_rejected(artifact):
    artifact['result']['split']='test'
    with pytest.raises(ValueError,match='train only'):collector.label_student_history(artifact)


def test_same_real_teacher_relabel_is_deterministic(artifact):
    first=collector.label_student_history(artifact)
    assert first==collector.label_student_history(copy.deepcopy(artifact))
    for sample in first[0]:
        frame=sample['view']['observations'][-1]
        if frame['remaining_budget']<4:assert sample['action']==0
