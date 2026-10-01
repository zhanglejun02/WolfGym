from dataclasses import asdict
import json

import pytest

from wolfgym.benchmark.combination_extensions import (
    BLOCKS, STRIDE, extension_contract, extension_manifest, extension_scenario,
    check_seed_namespaces, historical_inventory,
)
from wolfgym.env.combination_contract import validate_primary_scenario


@pytest.mark.parametrize('name',['liquidity','benign_spread'])
def test_exact_one_field_change_and_primary_entry_stays_locked(name):
    original = asdict(validate_primary_scenario())
    scenario = extension_scenario(name)
    changed = asdict(scenario)
    if name == 'liquidity':
        target = next(a for a in changed['assets'] if a['id']=='asset_2')
        assert target['daily_execution_liquidity'] == 9000
        target['daily_execution_liquidity'] = 12000
    else:
        assert changed['retail']['post_probability'] == .3
        assert changed['retail']['post_intensity'] == .2
        changed['retail']['post_probability'] = .1
    assert changed == original == asdict(validate_primary_scenario())
    with pytest.raises(ValueError,match='cannot be overridden'):
        validate_primary_scenario(scenario)


def test_complete_predeclared_conditions_and_disjoint_stages():
    all_roots = set()
    for name in BLOCKS:
        c = extension_contract(name)
        rows = extension_manifest(name)
        assert rows == extension_manifest(name)
        assert c['runtime_release']=='closed' and not c['primary_score_inclusion']
        assert c['learned_checkpoints']=={'11':1000,'22':2000,'33':1000}
        assert c['primary_task']['population']['harmful_fraction'] == .02
        for stage,count in [('development',16),('confirmation',128)]:
            r=[e for e in rows if e['stage']==stage]
            roots={e['trajectory_seed'] for e in r}
            assert len(roots)==count and not roots & all_roots
            all_roots |= roots
            assert len(r)==count*len(c['conditions'])
            for seed in roots:
                same=[e for e in r if e['trajectory_seed']==seed]
                assert {e['condition'] for e in same}==set(c['conditions'])
                assert len({e['group_id'] for e in same})==1
        assert c['condition_weights']==({'normal':.5,'attack':.5} if name=='liquidity' else {'normal':1.})
    roots={f'{n}/{s}':[start+STRIDE*i for i in range(count)] for n,b in BLOCKS.items() for s,(start,count) in b.items()}
    result=check_seed_namespaces(roots,[(0,53000255)])
    assert result['root_count']==288 and result['direct_entropy_values_checked']==2304


@pytest.mark.parametrize('roots,reserved',[
    ({'a':[100000000],'b':[101000000]},[]), # continuation input versus another root
    ({'a':[100000000,100000007]},[]), # market initializer versus another root
    ({'a':[100000000]},[(101999999,102000001)]), # replica overlaps retired range
    ({'a':[True]},[]),
])
def test_derived_not_just_root_collisions_are_rejected(roots,reserved):
    with pytest.raises(ValueError):check_seed_namespaces(roots,reserved)


def test_history_inventory_reads_metadata_and_filenames_without_outcomes(tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'data/study').mkdir(parents=True)
    (tmp_path/'configs/a.yaml').write_text('development: {seed_start: 71000, count: 16}\ncheckpoints: {11: 1000}\n')
    (tmp_path/'data/study/manifest.json').write_text(json.dumps({'seeds':[74000,74255]}))
    # Invalid gzip deliberately: the inventory must not open raw account data.
    (tmp_path/'data/study/75000_reference.json.gz').write_bytes(b'not a gzip')
    audit=historical_inventory(tmp_path)
    assert {71000,74000,74255,75000} <= set(audit['seed_numeric_values'])
    assert audit['raw_seed_filename_count']==1
    assert len(audit['metadata_sha256'])==2
    assert audit['reserved_intervals_inclusive'][-1]==[71000,3075000]


def test_unknown_extension_rejected():
    with pytest.raises(ValueError):extension_contract('stronger_attack')
