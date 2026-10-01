import copy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_combination_bc_round2 import summarize, BRANCHES
from wolfgym.evaluation.combination_host import read_json


@pytest.fixture
def grid():
    first=ROOT/'data/combination_bc_round1_20260927'
    if not (first/'acceptance.json').exists():pytest.skip('audited first-round fixture unavailable')
    rows=read_json(first/'audited_rows.json.gz')
    episodes=read_json(first/'manifest.json')['monitor']
    for branch,delta in zip(BRANCHES,(.0001,.0002)):
        for old in list(rows):
            if old['policy_id']=='bc_unweighted':
                new=copy.deepcopy(old);new['policy_id']=branch;new['benefit']+=delta;rows.append(new)
    return rows,episodes,['bc_unweighted','bc_balanced_wait_freeze','alarm_refresh',*BRANCHES]


def test_full_paired_difference(grid):
    results,paired=summarize(*grid)
    assert len(results)==5 and len(paired)==64
    assert all(r['aggregated_minus_fixed_pp']==pytest.approx(.01) for r in paired)


def test_missing_and_unpaired_rejected(grid):
    rows,episodes,policies=grid
    with pytest.raises(ValueError,match='comparison grid'):summarize(rows[:-1],episodes,policies)
    rows[-1]['reference_wait_sha256']='changed'
    with pytest.raises(ValueError,match='unpaired'):summarize(rows,episodes,policies)
