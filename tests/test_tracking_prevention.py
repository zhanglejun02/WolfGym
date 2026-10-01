from dataclasses import replace
import pytest
from test_task_validation_v4 import fixture
from wolfgym.evaluation.task_validation import public_observation
from wolfgym.evaluation.tracking_prevention import TrackingCampaignFreeze
from wolfgym.observations.schema import PublicMessageObservation


def test_target_is_refreshed_and_delay_does_not_access_future_or_require_new_alarm():
    obs = public_observation(fixture(), 12)
    a, b, _ = obs.assets
    def view(day, target, budget=12):
        return replace(obs, day=day, remaining_budget=budget,
                       messages=(PublicMessageObservation(day-1, 'visible', target, 'post', 'text', .8),))
    policy = TrackingCampaignFreeze(.7, 4)
    assert policy(view(1, a.asset_id), .8).action_type.name == 'WAIT'
    assert policy(view(4, b.asset_id), 0.).action_type.name == 'WAIT'
    assert policy(view(5, b.asset_id), 0.).target_id == b.asset_id
    assert policy(view(10, a.asset_id), 0.).target_id == a.asset_id
    assert policy(view(15, b.asset_id, 3), 0.).action_type.name == 'WAIT'
    assert policy(view(20, b.asset_id), 1.).action_type.name == 'WAIT'
    with pytest.raises(ValueError):
        TrackingCampaignFreeze(.7, 3)


def test_no_alarm_means_no_action_even_with_large_visible_reach():
    obs = replace(public_observation(fixture(), 12), messages=())
    policy = TrackingCampaignFreeze(.7)
    for day in range(30):
        assert policy(replace(obs, day=day), .7).action_type.name == 'WAIT'
