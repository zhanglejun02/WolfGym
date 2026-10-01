from dataclasses import replace
from test_counterfactual_welfare import histories
from test_task_validation_v4 import fixture, config
from wolfgym.evaluation.combination_task import score_combination
from wolfgym.evaluation.task_validation import run_validation_branch, public_features, public_observation
from wolfgym.evaluation.prevention_policy import PublicCampaignFreeze


def rows(n, blocked=0.):
    return [{'day': day, 'financial_outcomes': {'requested_notional': 100., 'blocked_notional': blocked,
             'executed_notional': 100. - blocked, 'unexecuted_notional': 0.}} for day in range(1, n + 1)]


def test_each_union_component_can_trigger_independently_and_recovery_does_not_erase():
    r, a = histories([[99.5] * 100, [99.5] * 100, [100.] * 100])
    out = score_combination(r, a, rows(3), rows(3), .0038562228090846347)
    assert out['components'] == {'wealth': True, 'tail': False, 'function': False}
    assert out['first_event_day'] == 2
    # Individual tail harm must not be hidden by gains elsewhere (G=0).
    r, a = histories([[90., 110.] + [100.] * 98])
    out = score_combination(r, a, rows(1), rows(1), .0038562228090846347)
    assert out['components'] == {'wealth': False, 'tail': True, 'function': False}
    r, a = histories([[100.] * 100, [100.] * 100])
    out = score_combination(r, a, rows(2), rows(2, 50.), .0038562228090846347)
    assert out['components'] == {'wealth': False, 'tail': False, 'function': True}
    assert not score_combination(r, a, rows(2), rows(2), .0038562228090846347)['event']


def test_shared_campaign_accounts_and_replica_control_pair_without_actions():
    sim = fixture()
    names = sorted(public_features(public_observation(sim, 12)))
    model = {'features': names, 'center': [0.] * len(names), 'scale': [1.] * len(names),
             'coefficient': [0.] * len(names), 'intercept': -30.}
    args = dict(seed=31, config=config(), pair_id='v5-test', model=model,
                frozen={'harm_threshold': .0038562228090846347, 'alarm_threshold': .7})
    initial = sim.get_state()
    r, _ = run_validation_branch(sim, initial, name='normal_replica', **args)
    a, days = run_validation_branch(sim, initial, name='normal_public', normal_decision_replica=True,
                                   policy_factory=lambda: PublicCampaignFreeze(.7), **args)
    assert r == a and sum(row['cost'] for row in days) == 0
    a, days = run_validation_branch(sim, initial, name='attack', tactic_mode='campaign', **args)
    assert len({row['choice']['executed'] for row in days if row['tactic_activity']['orders'] or row['tactic_activity']['messages']}) >= 2
    assert (a, days) == run_validation_branch(sim, initial, name='attack', tactic_mode='campaign', **args)
