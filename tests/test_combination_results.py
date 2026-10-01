"""Synthetic arithmetic fixtures only: no market or model evaluation."""
import copy

import pytest

from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.evaluation.combination_results import paired_results


def fixture():
    episodes = [CombinationEpisode('train', 30000000+g, c).to_dict()
                for g in range(4) for c in ('normal', 'attack')]
    members = {'rule': ['fixed'], 'learned': ['11', '22', '33']}
    rows = []
    for policy, seeds in members.items():
        for member in seeds:
            for episode in episodes:
                g = episode['trajectory_seed']-30000000
                # Large opposite condition effects cancel within each society.
                sign = 1 if episode['condition'] == 'attack' else -1
                offset = {'fixed': 0, '11': -.003, '22': 0, '33': .003}[member]
                benefit = sign * (.02 + .01*g) + (0 if policy == 'rule' else .001) + offset
                def score(h):
                    return {'horizon_complete': True, 'damage': {'H': [h]*30, 'G': [-.002]*30},
                            'event': False, 'components': dict(wealth=False, tail=False, function=False)}
                financial = dict(requested_notional=100., blocked_notional=10., unexecuted_notional=20., executed_notional=70.)
                result = {**episode, 'H_reduction': benefit, 'wait': score(.1), 'policy': score(.1-benefit),
                          'cost': 4., 'invalid_actions': 0, 'model_decisions': 0,
                          'financial_outcomes': {b: dict(financial) for b in ('reference', 'wait', 'policy')}}
                rows.append({'policy_id': policy, 'member_id': member, 'result': result})
    return dict(rows=rows, expected_episodes=episodes, policy_members=members, population='development_subset')


def test_signed_benefits_conditions_and_seed_replicates_stay_in_society():
    report = paired_results(**fixture())
    assert report['independent_societies'] == 4 and report['result_count'] == 32
    learned = report['policies']['learned']
    balanced = learned['summary']['balanced']['benefit_pp']
    assert balanced['mean'] == pytest.approx(.1)
    assert balanced['interval'] == pytest.approx([.1, .1])
    assert learned['summary']['normal']['benefit_pp']['mean'] < 0
    assert learned['summary']['attack']['benefit_pp']['mean'] > 0
    assert learned['summary']['normal']['G_policy_pp']['mean'] == pytest.approx(-.2)
    assert learned['summary']['balanced']['invalid_action_rate'] is None
    assert [learned['members'][m]['balanced']['mean'] for m in ('11','22','33')] == pytest.approx([-.2, .1, .4])
    assert not report['official_release_authorized']


def test_paired_comparisons_keep_common_shocks_and_full_family():
    data = fixture()
    base = [r for r in data['rows'] if r['policy_id'] == 'rule']
    data['rows'] = list(base)
    data['policy_members'] = {'rule': ['fixed']}
    for i in range(1, 5):
        p = f'M{i}'
        data['policy_members'][p] = ['fixed']
        for row in base:
            row = copy.deepcopy(row)
            row['policy_id'] = p
            # Same society variation in both policies; paired contrast constant.
            row['result']['H_reduction'] += i*.001
            row['result']['policy']['damage']['H'] = [h-i*.001 for h in row['result']['policy']['damage']['H']]
            data['rows'].append(row)
    data['comparison_families'] = {'main': [dict(id=f'M{i}-rule', left=f'M{i}', right='rule') for i in range(1, 5)]}
    report = paired_results(**data)
    for i in range(1, 5):
        diff = report['comparisons']['main'][f'M{i}-rule']
        assert diff['raw']['interval'] == pytest.approx([i*.1, i*.1])
        assert diff['bonferroni']['confidence'] == .9875
    data['comparison_families']['main'].pop()
    with pytest.raises(ValueError, match='K=4/8'):
        paired_results(**data)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'foreign_member', 'failed_horizon', 'identity', 'nan', 'wrong_benefit', 'budget', 'invalid_count', 'financial', 'or_event', 'wait_drift'])
def test_invalid_or_incomplete_results_cannot_produce_subset_table(change):
    data = fixture()
    result = data['rows'][0]['result']
    if change == 'missing': data['rows'].pop()
    elif change == 'duplicate': data['rows'].append(copy.deepcopy(data['rows'][0]))
    elif change == 'foreign_member': data['rows'][0]['member_id'] = 'best_seed'
    elif change == 'failed_horizon': result['policy']['horizon_complete'] = False
    elif change == 'identity': result['trajectory_seed'] += 1
    elif change == 'nan': result['policy']['damage']['H'][0] = float('nan')
    elif change == 'wrong_benefit': result['H_reduction'] = 0
    elif change == 'budget': result['cost'] = 16
    elif change == 'invalid_count': result['invalid_actions'] = 1
    elif change == 'financial': result['financial_outcomes']['policy']['executed_notional'] = 99
    elif change == 'or_event': result['policy']['event'] = True
    elif change == 'wait_drift': result['wait']['damage']['G'][0] = -.001
    with pytest.raises(ValueError): paired_results(**data)


@pytest.mark.parametrize('split,start,population', [('validation',40000000,'development_subset'), ('test',50000000,'development_subset'), ('test',50000000,'full_split')])
def test_cannot_call_small_validation_or_test_subset_complete(split, start, population):
    data = fixture()
    data['expected_episodes'] = [CombinationEpisode(split, start, c).to_dict() for c in ('normal','attack')]
    data['population'] = population
    with pytest.raises(ValueError): paired_results(**data)


def test_population_requires_both_conditions_and_unique_identities():
    data = fixture()
    data['expected_episodes'].pop()
    with pytest.raises(ValueError, match='paired'): paired_results(**data)
    data = fixture()
    data['expected_episodes'].append(data['expected_episodes'][0])
    with pytest.raises(ValueError, match='duplicate'): paired_results(**data)


def test_event_function_and_global_invalid_rate_are_reported_separately():
    data = fixture()
    for row in data['rows']:
        r = row['result']
        if row['policy_id'] == 'rule':
            r['policy']['components']['function'] = r['policy']['event'] = True
            r['financial_outcomes']['policy'].update(blocked_notional=20., executed_notional=60.)
            r['invalid_actions'] = 1 if r['condition'] == 'normal' else 0
            r['model_decisions'] = 1 if r['condition'] == 'normal' else 9
    report = paired_results(**data)['policies']['rule']['summary']
    assert report['balanced']['invalid_action_rate'] == .1  # ratio of totals, not average rates
    assert report['normal']['new_event']['mean'] == 1
    assert report['normal']['function_new']['mean'] == 1
    assert report['normal']['wealth_new']['mean'] == 0
    assert report['normal']['executed_notional_delta']['mean'] == -10


def write_completion(directory):
    """Synthetic artifact, deliberately no simulated accounts or paid ledger."""
    import gzip
    import hashlib
    import json
    result = copy.deepcopy(fixture()['rows'][0]['result'])
    episode = CombinationEpisode('engineering', 20260925, 'normal').to_dict()
    result.update(episode)
    result['cost'] = 0
    manifest = dict(version='combination-model-engineering-v1', episode=episode, formal_test_allowed=False)
    scope = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (directory/'manifest.json').write_text(json.dumps(manifest))
    (directory/'decisions').mkdir()
    decisions = [dict(day=d, action=0, invalid_action=False, request=None) for d in range(30)]
    for row in decisions:
        (directory/f"decisions/{row['day']:02d}.json").write_text(json.dumps(row))
    with gzip.open(directory/'outcome.json.gz','wt') as f:
        json.dump(dict(scope=scope,result=result,decisions=decisions), f)
    receipt = dict(scope=scope, sha256={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in directory.rglob('*') if p.is_file()})
    (directory/'complete.json').write_text(json.dumps(receipt))
    return receipt


def test_complete_artifact_reader_and_missing_day_receipt(tmp_path):
    import json
    from wolfgym.evaluation.combination_results import completed_engineering_result
    receipt = write_completion(tmp_path)
    assert completed_engineering_result(tmp_path)['split'] == 'engineering'
    del receipt['sha256']['decisions/29.json']
    (tmp_path/'complete.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='30 decisions'):
        completed_engineering_result(tmp_path)


def test_completed_artifact_corruption_is_rejected(tmp_path):
    from wolfgym.evaluation.combination_results import completed_engineering_result
    write_completion(tmp_path)
    (tmp_path/'decisions/00.json').write_text('{}')
    with pytest.raises(ValueError, match='fingerprint'):
        completed_engineering_result(tmp_path)


def test_hash_consistent_but_wrong_terminal_action_counts_rejected(tmp_path):
    import gzip
    import hashlib
    import json
    from wolfgym.evaluation.combination_results import completed_engineering_result
    receipt = write_completion(tmp_path)
    with gzip.open(tmp_path/'outcome.json.gz', 'rt') as f:
        outcome = json.load(f)
    outcome['result']['cost'] = 4
    with gzip.open(tmp_path/'outcome.json.gz', 'wt') as f:
        json.dump(outcome, f)
    receipt['sha256']['outcome.json.gz'] = hashlib.sha256((tmp_path/'outcome.json.gz').read_bytes()).hexdigest()
    (tmp_path/'complete.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='counts'):
        completed_engineering_result(tmp_path)


def test_row_order_does_not_change_paired_statistics_or_declaration():
    data = fixture()
    first = paired_results(**data)
    data['rows'].reverse()
    data['expected_episodes'].reverse()
    assert paired_results(**data) == first


def test_ability_family_keeps_eight_distinct_contrasts():
    data = fixture()
    originals = [copy.deepcopy(r) for r in data['rows'] if r['policy_id'] == 'rule']
    for p in ('RL1', 'LR1', 'LL1', 'C1', 'RL3', 'LR3', 'LL3', 'C3'):
        data['policy_members'][p] = ['fixed']
        for r in originals:
            r = copy.deepcopy(r)
            r['policy_id'] = p
            data['rows'].append(r)
    declarations = []
    for m in ('1','3'):
        for left, right in [(f'RL{m}','rule'),(f'LR{m}','rule'),(f'LL{m}',f'RL{m}'),(f'LL{m}',f'C{m}')]:
            declarations.append(dict(id=left+'-'+right,left=left,right=right))
    data['comparison_families'] = {'ability': declarations}
    comparisons = paired_results(**data)['comparisons']['ability']
    assert len(comparisons) == 8
    assert all(c['bonferroni']['confidence'] == .99375 for c in comparisons.values())
    declarations[-1] = dict(declarations[0])
    with pytest.raises(ValueError, match='duplicate'): paired_results(**data)
