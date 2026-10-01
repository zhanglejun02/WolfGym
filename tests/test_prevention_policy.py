from dataclasses import replace
import pytest

from wolfgym.evaluation.prevention_policy import PublicCampaignFreeze
from wolfgym.evaluation.task_validation import public_observation, run_validation_branch
from wolfgym.observations.schema import PublicMessageObservation
from wolfgym.regulation import RegulatoryActionType
from test_task_validation_v4 import fixture, config


def test_public_target_and_three_spaced_one_day_freezes():
    sim = fixture()
    obs = public_observation(sim, 12)
    a, b, _ = obs.assets
    obs = replace(obs, messages=(PublicMessageObservation(2, "public-person", a.asset_id, "post", "text", .2),
                                 PublicMessageObservation(2, "another-person", b.asset_id, "post", "text", .5)))
    policy = PublicCampaignFreeze(.7)
    assert policy(replace(obs, day=2), .69).action_type is RegulatoryActionType.WAIT
    for day in range(3, 20):
        # Later messages and scores cannot secretly change the latched schedule.
        action = policy(replace(obs, day=day, remaining_budget=12), .8 if day == 3 else 0.)
        assert (action.action_type is RegulatoryActionType.FREEZE) == (day in (3, 8, 13))
        if day in (3, 8, 13):
            assert action.target_id == b.asset_id
    assert policy(replace(obs, day=13, remaining_budget=3), .9).action_type is RegulatoryActionType.WAIT


def test_missing_public_messages_rejected_and_empty_messages_use_public_fallback():
    obs = public_observation(fixture(), 12)
    with pytest.raises(ValueError):
        PublicCampaignFreeze(.7)(obs, .8)
    target = max(obs.assets, key=lambda a: (a.return_3d, a.volume_zscore, a.asset_id)).asset_id
    assert PublicCampaignFreeze(.7)(replace(obs, messages=()), .8).target_id == target


def test_policy_hook_is_fresh_on_replay_and_controls_release():
    sim = fixture()
    observation = public_observation(sim, 12)
    from wolfgym.evaluation.task_validation import public_features
    names = sorted(public_features(observation))
    model = {"features": names, "center": [0.] * len(names), "scale": [1.] * len(names),
             "coefficient": [0.] * len(names), "intercept": 10.}
    args = dict(seed=31, name="attack_public", config=config(), pair_id="fixture", model=model,
                frozen={"alarm_threshold": .5, "harm_threshold": .0025}, policy_factory=lambda: PublicCampaignFreeze(.5))
    initial = sim.get_state()
    snapshots, rows = run_validation_branch(sim, initial, **args)
    assert (snapshots, rows) == run_validation_branch(sim, initial, **args)
    assert [r["day"] for r in rows if r["cost"]] == [1, 6, 11]
    assert sum(r["cost"] for r in rows) == 12
    assert all(r["financial_outcomes"]["blocked_notional"] == 0 for r in rows if not r["cost"])
