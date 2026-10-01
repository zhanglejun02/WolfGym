"""Adversarial checks using one saved train-monitor episode, no new simulation."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_combination_bc_pilot import audit_one, summarize
from wolfgym.evaluation.combination_host import read_json, file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest


@pytest.fixture
def saved(tmp_path):
    source=ROOT/'data/combination_bc_round1_20260927'
    if not (source/'manifest.json').exists():pytest.skip('integration fixture unavailable')
    manifest=read_json(source/'manifest.json');identity=manifest['monitor'][0]
    policy='bc_unweighted';name=identity['episode_id'].replace(':','_')
    artifact=source/'monitor'/policy/name/'artifact.json.gz'
    if not artifact.exists():pytest.skip('integration episode not yet committed')
    directory=tmp_path/'monitor'/policy/name;directory.mkdir(parents=True)
    (tmp_path/policy).mkdir()
    (tmp_path/policy/'model_1000.pt').write_bytes((source/policy/'model_1000.pt').read_bytes())
    a=read_json(artifact);execution=digest(manifest)
    return tmp_path,directory,a,(str(tmp_path),identity,policy,execution)


def commit(directory,a,execution):
    durable_json(directory/'artifact.json.gz',a)
    durable_json(directory/'complete.json',{'execution_id':execution,'sha256':file_hash(directory/'artifact.json.gz')})


def test_valid_saved_account_and_model_replay(saved):
    _,d,a,job=saved;commit(d,a,job[-1]);result=audit_one(job)
    assert result['episode_id']==job[1]['episode_id']


def test_rehashed_score_corruption_rejected(saved):
    _,d,a,job=saved;a['result']['H_reduction']+=.01;commit(d,a,job[-1])
    with pytest.raises(ValueError,match='benefit mismatch'):audit_one(job)


def test_rehashed_budget_corruption_rejected(saved):
    _,d,a,job=saved;a['decisions'][2]['frame']['remaining_budget']=-4;commit(d,a,job[-1])
    with pytest.raises(ValueError,match='budget trace mismatch'):audit_one(job)


def test_partial_grid_rejected():
    with pytest.raises(ValueError,match='incomplete monitor grid'):summarize([])
