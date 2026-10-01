"""Counterfactual social randomness must be invariant to message filtering."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from wolfgym.core.agents.social_game import BoundedSocialEnv
from wolfgym.core.env.social import Message, SocialEnv, SocialGraph


def _graph() -> SocialGraph:
    graph = SocialGraph(12, 6, np.random.default_rng(17))
    graph.assign_ids([f"agent_{index}" for index in range(12)])
    return graph


def _scenario(*, shuffle_sender: bool = False) -> SimpleNamespace:
    return SimpleNamespace(social={
        "feedback_strength": 0.6,
        "message_decay": 0.5,
        "bot_ratio": 0.05,
        "p_expose": 0.5,
        "p_reshare": 0.5,
        "content_visible": True,
        "social_proof_visible": True,
        "shuffle_sender": shuffle_sender,
    })


def _messages() -> tuple[Message, Message]:
    harmful = Message(
        "agent_0", "asset_2", 1.0, 2.0,
        is_harmful=True, is_bot=True, day=4, text="Coordinated promotion.",
    )
    normal = Message(
        "agent_1", "asset_2", -1.0, 0.8,
        day=4, text="Ordinary skeptical view.",
    )
    return harmful, normal


def _normal_message_event(env) -> dict:
    return next(
        event for event in env.message_events
        if event["sender_id"] == "agent_1"
    )


def test_base_social_audience_is_invariant_to_filtering_another_message() -> None:
    harmful, normal = _messages()
    full = SocialEnv(_graph(), _scenario(), np.random.default_rng(91))
    filtered = SocialEnv(_graph(), _scenario(), np.random.default_rng(91))

    full.step(4, [harmful, normal], {"asset_2": 0.01})
    filtered.step(4, [normal], {"asset_2": 0.01})

    assert _normal_message_event(full) == _normal_message_event(filtered)


def test_bounded_social_normal_deliveries_and_sender_mask_are_crn_stable() -> None:
    harmful, normal = _messages()
    full = BoundedSocialEnv(
        _graph(), _scenario(shuffle_sender=True), np.random.default_rng(123),
    )
    filtered = BoundedSocialEnv(
        _graph(), _scenario(shuffle_sender=True), np.random.default_rng(123),
    )

    full.step(4, [harmful, normal], {"asset_2": 0.0})
    filtered.step(4, [normal], {"asset_2": 0.0})

    normal_id = _normal_message_event(full)["message_id"]
    full_deliveries = [
        event for event in full.exposure_events
        if event["message_id"] == normal_id
    ]
    filtered_deliveries = [
        event for event in filtered.exposure_events
        if event["message_id"] == normal_id
    ]
    assert _normal_message_event(full) == _normal_message_event(filtered)
    assert full_deliveries == filtered_deliveries


def test_intensity_scaling_preserves_the_message_randomness_identity() -> None:
    _, normal = _messages()
    scaled = Message(
        sender_id=normal.sender_id,
        asset=normal.asset,
        sentiment=normal.sentiment,
        intensity=normal.intensity * 0.4,
        day=normal.day,
        kind=normal.kind,
        confidence=normal.confidence,
        text=normal.text,
    )
    assert SocialEnv._message_id(4, normal) == SocialEnv._message_id(4, scaled)
