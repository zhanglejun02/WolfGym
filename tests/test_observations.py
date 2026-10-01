from __future__ import annotations

import json
import inspect
import math
import re

import pytest

from wolfgym.core import WolfSocietySimulator
from wolfgym.core.env.social import Message
from wolfgym.core.scenarios.base import load_scenario
from wolfgym.counterfactual import CounterfactualEvaluator
from wolfgym.observations import (
    EntityObservation,
    FORBIDDEN_FIELDS,
    InformationLevel,
    MarketObservation,
    ObservationBuilder,
    ObservationConfig,
    OBSERVATION_SCHEMA_VERSION,
    LANGUAGE_OBSERVATION_SCHEMA_VERSION,
    PublicMessageObservation,
    RegulatoryObservation,
    SocialObservation,
)
from wolfgym.observations.serialization import assert_no_hidden_fields
from wolfgym.regulation import RegulatoryAction, RegulatoryActionType


def _state(scenario: str = "s1", days: int = 8):
    env = WolfSocietySimulator(
        load_scenario(scenario),
        n_society=40,
        alpha=0.10,
        seed=17,
    )
    for _ in range(days):
        env.step_day()
    return env, env.get_state()


def test_observation_is_deterministic_and_does_not_mutate_simulator() -> None:
    env, state = _state()
    builder = ObservationBuilder()
    before_day = env._next_day
    first = builder.build(env, state)
    second = builder.build(env, state)
    assert first == second
    assert env._next_day == before_day

    env.step_day()
    assert builder.build(env, state) == first


def test_default_observation_is_small_serializable_and_finite() -> None:
    env, state = _state()
    observation = ObservationBuilder().build(env, state)
    payload = observation.to_dict()
    assert observation.schema_version == OBSERVATION_SCHEMA_VERSION
    assert "messages" not in payload
    assert json.loads(observation.to_prompt()) == payload
    assert payload["information_level"] == 2
    assert payload["society_size"] == 40
    assert len(observation.entities) <= 5
    assert len(observation.assets) == len(env.market.assets)
    assert set(payload["market"]) == {
        "return_1d", "return_3d", "volatility_3d", "volume_zscore",
        "liquidity", "order_imbalance", "cancellation_rate",
    }
    assert set(payload["social"]) == {
        "message_rate", "cascade_reach", "sentiment_concentration",
        "network_degree_concentration", "information_concentration",
        "consensus_concentration",
    }
    numeric_values = [*payload["market"].values(), *payload["social"].values()]
    assert all(math.isfinite(value) for value in numeric_values)
    assert abs(observation.market.volume_zscore) <= 10.0
    text = observation.to_text()
    assert text.startswith(f"Day {observation.day}.")
    assert "Market conditions:" in text
    assert "Society size: 40 agents." in text
    assert "Observed assets:" in text
    assert "Observed entities:" in text
    assert "Recent public messages:" not in text


def test_language_observation_is_versioned_public_and_deterministic() -> None:
    env, state = _state("s1", days=10)
    builder = ObservationBuilder(ObservationConfig(
        include_messages=True,
        max_messages=6,
    ))
    first = builder.build(env, state)
    second = builder.build(env, state)

    assert first == second
    assert first.schema_version == LANGUAGE_OBSERVATION_SCHEMA_VERSION
    assert first.messages is not None
    assert first.messages
    assert len(first.messages) <= 6
    payload = first.to_dict()
    assert payload["schema_version"] == LANGUAGE_OBSERVATION_SCHEMA_VERSION
    assert payload["messages"]
    assert "Recent public messages:" in first.to_text()

    prompt = first.to_prompt()
    for internal_id in [agent.agent_id for agent in env.society.all_agents]:
        assert internal_id not in prompt
    for internal_id in env.market.assets:
        assert all(internal_id not in message.text for message in first.messages)
    for message in first.messages:
        assert re.fullmatch(r"entity_[0-9a-f]{12}", message.sender_id)
        assert re.fullmatch(r"asset_[0-9a-f]{12}", message.asset_id)
        assert message.text.strip()
        assert 0.0 <= message.audience_reach <= 1.0


def test_public_speech_is_not_mistaken_for_hidden_metadata() -> None:
    observation = RegulatoryObservation(
        day=1,
        society_size=40,
        information_level=InformationLevel.TOP_ENTITIES,
        market=MarketObservation(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        social=SocialObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        messages=(PublicMessageObservation(
            day=0,
            sender_id="entity_000000000001",
            asset_id="asset_000000000001",
            kind="post",
            text="I think system failure is possible if liquidity keeps falling.",
            audience_reach=0.25,
        ),),
    )
    assert observation.to_dict()["messages"][0]["text"].startswith("I think")

    with pytest.raises(ValueError, match="forbidden observation field"):
        assert_no_hidden_fields({"messages": [{"text": "public", "oracle_risk": 0.2}]})


def test_rule_language_does_not_depend_on_hidden_source_status() -> None:
    public = Message("retail_1", "asset_1", 1.0, 1.0, day=4)
    hidden = Message(
        "retail_1", "asset_1", 1.0, 1.0, is_harmful=True, is_bot=True, day=4,
    )
    assert public.text == hidden.text


def test_raw_identifiers_inside_public_text_are_pseudonymized() -> None:
    env, _ = _state("s1", days=10)
    event = env.social.message_events[-1]
    raw_sender = str(event["sender_id"])
    raw_asset = str(event["asset"])
    event["text"] = f"{raw_sender} is discussing {raw_asset}."
    observation = ObservationBuilder(ObservationConfig(
        include_messages=True,
        max_messages=8,
    )).build(env, env.get_state())
    rendered = " ".join(message.text for message in observation.messages or ())
    assert not re.search(rf"\b{re.escape(raw_sender)}\b", rendered)
    assert not re.search(rf"\b{re.escape(raw_asset)}\b", rendered)
    assert env.public_entity_id(raw_sender) in rendered
    assert env.public_asset_id(raw_asset) in rendered


def test_language_events_are_part_of_cloned_state_without_polluting_source() -> None:
    env, state = _state("s1", days=10)
    before = ObservationBuilder(ObservationConfig(include_messages=True)).build(env, state)
    source_event_count = len(env.social.message_events)

    branch = env.fork_from_state(state)
    branch.step_day()
    assert len(branch.social.message_events) >= source_event_count
    assert len(env.social.message_events) == source_event_count
    assert ObservationBuilder(ObservationConfig(include_messages=True)).build(env, state) == before


def test_entity_ids_are_opaque_and_raw_role_ids_never_serialize() -> None:
    env, state = _state("s3", days=4)
    observation = ObservationBuilder().build(env, state)
    prompt = observation.to_prompt()
    assert observation.entities
    assert all(re.fullmatch(r"entity_[0-9a-f]{12}", item.entity_id) for item in observation.entities)
    assert any(item.cancellation_rate > 0.0 for item in observation.entities)
    for agent in env.society.all_agents:
        assert agent.agent_id not in prompt
    assert not any(field in prompt.casefold() for field in FORBIDDEN_FIELDS)


def test_information_levels_gate_entities_and_investigation() -> None:
    env, state = _state()
    aggregate = ObservationBuilder(ObservationConfig(
        information_level=InformationLevel.AGGREGATE,
    )).build(env, state)
    assert aggregate.entities == ()
    assert aggregate.investigation is None

    entity_builder = ObservationBuilder(ObservationConfig(
        information_level=InformationLevel.TOP_ENTITIES,
        top_k_entities=3,
    ))
    entity_view = entity_builder.build(env, state)
    assert 0 < len(entity_view.entities) <= 3
    target = entity_view.entities[0].entity_id
    with pytest.raises(ValueError, match="level 3"):
        entity_builder.build(env, state, investigation_target=target)

    investigated = ObservationBuilder(ObservationConfig(
        information_level=InformationLevel.INVESTIGATION,
        top_k_entities=3,
    )).build(env, state, investigation_target=target)
    assert investigated.investigation is not None
    assert investigated.investigation.target_id == target
    assert len(investigated.investigation.recent_message_pattern) <= 3
    assert len(investigated.investigation.recent_order_pattern) <= 3
    assert all(
        re.fullmatch(r"entity_[0-9a-f]{12}", neighbour)
        for neighbour in investigated.investigation.network_neighbors
    )


def test_leakage_guard_checks_structured_fields_and_final_values() -> None:
    with pytest.raises(ValueError, match="forbidden observation field"):
        assert_no_hidden_fields({"oracle_action": "WAIT"})

    contaminated = RegulatoryObservation(
        day=1,
        society_size=40,
        information_level=InformationLevel.TOP_ENTITIES,
        market=MarketObservation(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        social=SocialObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        entities=(EntityObservation("attacker_1", 0.0, 0.0, 0.0, 0.0, 0.0),),
    )
    with pytest.raises(ValueError, match="forbidden observation value"):
        contaminated.to_prompt()


def test_social_and_entity_features_come_from_public_activity() -> None:
    env, state = _state("s1", days=10)
    observation = ObservationBuilder().build(env, state)
    assert observation.social.message_rate > 0.0
    assert 0.0 <= observation.social.cascade_reach <= 1.0
    assert all(0.0 <= entity.message_reach <= 1.0 for entity in observation.entities)
    assert all(0.0 <= entity.trading_concentration <= 1.0 for entity in observation.entities)
    assert all(0.0 <= entity.social_influence <= 1.0 for entity in observation.entities)


def test_market_volatility_is_mean_of_per_asset_temporal_volatilities() -> None:
    env, state = _state("s3", days=8)
    observation = ObservationBuilder().build(env, state)
    expected = sum(asset.volatility_3d for asset in observation.assets) / len(
        observation.assets
    )
    assert observation.market.volatility_3d == pytest.approx(expected)


def test_zero_to_positive_activity_is_a_clipped_spike() -> None:
    builder = ObservationBuilder(ObservationConfig(activity_change_clip=10.0))
    assert builder._activity_change(100.0, 0.0) == 10.0
    assert builder._activity_change(0.0, 0.0) == 1.0
    assert builder._activity_change(4.0, 2.0) == 2.0


def test_public_ids_are_episode_local_and_asset_ids_are_actionable() -> None:
    env, state = _state("s3", days=4)
    same = WolfSocietySimulator(load_scenario("s3"), 40, 0.10, seed=17)
    different = WolfSocietySimulator(load_scenario("s3"), 40, 0.05, seed=17)
    internal_entity = env.society.all_agents[0].agent_id
    internal_asset = next(iter(env.market.assets))
    assert env.public_entity_id(internal_entity) == same.public_entity_id(internal_entity)
    assert env.public_entity_id(internal_entity) != different.public_entity_id(internal_entity)

    public_asset = env.public_asset_id(internal_asset)
    assert re.fullmatch(r"asset_[0-9a-f]{12}", public_asset)
    assert env.internal_asset_id(public_asset) == internal_asset
    result = CounterfactualEvaluator().evaluate_state(
        env,
        state,
        [
            RegulatoryAction(RegulatoryActionType.WAIT),
            RegulatoryAction(RegulatoryActionType.THROTTLE, public_asset),
        ],
        seeds=[1],
        horizon=1,
    )
    assert f"THROTTLE:{public_asset}" in result.actions


def test_entity_telemetry_has_no_role_or_hidden_mechanism_inputs() -> None:
    env, _ = _state("s3", days=4)
    history = env.observable_entity_history()
    assert history
    allowed = {
        "message_intensity", "potential_message_reach", "order_quantity",
        "order_count", "cancellation_count",
    }
    assert all(set(stats) == allowed for day in history for stats in day["entities"].values())
    source = inspect.getsource(env._record_observable_entity_activity)
    assert "is_spoof" not in source
    assert "is_harmful" not in source
    assert "role" not in source


def test_observed_entity_id_can_be_used_as_counterfactual_action_target() -> None:
    env, state = _state("s3", days=4)
    observation = ObservationBuilder().build(env, state)
    target = max(observation.entities, key=lambda entity: entity.cancellation_rate).entity_id
    result = CounterfactualEvaluator().evaluate_state(
        simulator=env,
        state=state,
        candidate_actions=[
            RegulatoryAction(RegulatoryActionType.WAIT),
            RegulatoryAction(RegulatoryActionType.FREEZE, target),
        ],
        seeds=[1, 2],
        horizon=2,
    )
    assert f"FREEZE:{target}" in result.actions
    assert env._next_day == state.next_day
