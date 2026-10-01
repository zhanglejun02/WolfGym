import importlib.util
import gzip
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_pilot',ROOT/'scripts/prepare_combination_imitation_pilot.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def rows():
    return [{'result':{'condition':c,'cost':0,'H_reduction':0.},'actions':[0]*30} for c in ('normal','attack')]


def test_no_action_is_distinct_from_unavailable_action_timing():
    summary=pilot.summarize(rows())
    assert all(r['all_wait_episode_fraction']==1 and r['mean_first_freeze_day_if_any'] is None for r in summary.values())
    r=rows();r[1]['actions'][0:3]=[1,2,3];r[1]['result'].update(cost=12,H_reduction=-.001)
    result=pilot.summarize(r)['attack']
    assert result['mean_budget_exhausted_day_if_any']==2
    assert result['benefit_mean_pp']==-.1 and result['negative_benefit_fraction']==1


def test_budget_and_missing_condition_fail_closed():
    r=rows();r[0]['actions'][0]=1
    with pytest.raises(ValueError,match='budget'):pilot.summarize(r)
    with pytest.raises(ValueError,match='condition'):pilot.summarize(rows()[:1])


def test_prepared_public_examples_and_partition_boundaries():
    folder=ROOT/'data/combination_imitation_preparation_20260927'
    if not folder.exists():
        pytest.skip('private imitation-preparation artifacts are unavailable')
    provenance=json.loads((folder/'provenance.json').read_text())
    partitions=json.loads((folder/'partitions.json').read_text())
    groups={k:{e['group_id'] for e in v} for k,v in partitions.items()}
    assert [len(groups[k]) for k in ('demonstrations','development_monitor','student_collection')]==[64,64,128]
    assert sum(map(len,groups.values()))==len(set.union(*groups.values()))
    assert all(e['split']=='train' for eps in partitions.values() for e in eps)
    with gzip.open(folder/'demonstrations.jsonl.gz','rt') as f:
        samples=[json.loads(line) for line in f]
    assert len(samples)==len(provenance)==3840
    freezes=0
    for i,row in enumerate(samples):
        assert set(row)=={'sample_id','view','action'} and row['sample_id']==i
        assert provenance[i]['sample_id']==i
        assert set(row['view'])=={'schema_version','window','history_mask','observations','interventions'}
        frame=row['view']['observations'][-1]
        assert frame['action_mask'][row['action']]
        assert frame['day']==provenance[i]['day']
        assert not ({'condition','trajectory_seed','H','G','accounts'} & set(frame))
        freezes+=row['action']!=0
    assert freezes==192
    receipt=json.loads((folder/'preparation_receipt.json').read_text())
    assert all(pilot.file_hash(folder/p)==h for p,h in receipt['outputs_sha256'].items())
