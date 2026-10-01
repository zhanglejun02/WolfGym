from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_combination_bc_final import paired_arrays, interval, POLICIES
from wolfgym.benchmark.combination import episode_plan


@pytest.fixture
def grid():
    episodes=[e.to_dict() for e in episode_plan('validation')]
    records=[{**e,'policy_id':p,'benefit':(i+1)*.0001,
              'reference_wait_sha256':e['episode_id']} for i,p in enumerate(POLICIES) for e in episodes]
    return records,episodes


def test_three_seeds_average_within128_societies(grid):
    records,episodes=grid
    roots,values=paired_arrays(records,episodes,POLICIES)
    average=np.mean([values[p]['balanced'] for p in POLICIES],axis=0)
    assert len(roots)==128 and average.shape==(128,)
    assert np.allclose(average,.02)
    ci=interval(average,roots)
    assert ci['mean_pp']==pytest.approx(.02)
    assert ci['society_bootstrap_ci95_pp']==pytest.approx([.02,.02])
    assert ci['dependence_sensitivity']['chain_count']==13
    assert not ci['dependence_sensitivity']['coverage_certified']


def test_incomplete_or_duplicate_seed_grid_rejected(grid):
    records,episodes=grid
    with pytest.raises(ValueError,match='seed episode grid'):paired_arrays(records[:-1],episodes,POLICIES)
    records[-1]=records[0]
    with pytest.raises(ValueError,match='seed episode grid'):paired_arrays(records,episodes,POLICIES)


def test_reference_wait_pairing_rejected(grid):
    records,episodes=grid;records[0]['reference_wait_sha256']='changed'
    with pytest.raises(ValueError,match='unpaired current seeds'):paired_arrays(records,episodes,POLICIES)


def test_historical_log_metadata_ignored_but_financial_change_rejected(monkeypatch):
    import report_combination_bc_final as report
    from wolfgym.evaluation.combination_public import digest
    identity={'trajectory_seed':40000000,'condition':'normal','episode_id':'fixture'}
    cache={'episode':identity,'execution_id':'e','reference':[{'cash':1}], 'wait':[{'cash':2}],
           'reference_rows':[{'day':1,'financial_outcomes':{'executed_notional':10},'old_diagnostic':99}],
           'wait_rows':[{'day':1,'financial_outcomes':{'executed_notional':20},'old_diagnostic':99}]}
    marker={'episode':identity,'execution_id':'e','sha256':{'reference_wait.json.gz':'h'}}
    expected=digest({world:{'accounts':cache[world],
                    'rows':[{'day':1,'financial_outcomes':cache[world+'_rows'][0]['financial_outcomes']}]}
                    for world in ('reference','wait')})
    monkeypatch.setattr(report,'read_json',lambda p: marker if p.name=='complete.json' else cache)
    monkeypatch.setattr(report,'file_hash',lambda _: 'h')
    assert report.historical_pair((identity,expected))=='fixture'
    cache['wait_rows'][0]['financial_outcomes']['executed_notional']=21
    with pytest.raises(ValueError,match='reference/WAIT differ'):report.historical_pair((identity,expected))
