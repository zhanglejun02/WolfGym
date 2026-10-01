"""Leakage-checked observation and prompt serialization."""
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from wolfgym.observations.schema import RegulatoryObservation


FORBIDDEN_FIELDS = frozenset({
    "trajectory_id",
    "instance_id",
    "scenario_id",
    "hidden_state_ref",
    "trajectory_seed",
    "trusted_generation",
    "harmful",
    "malicious",
    "attacker",
    "attack_stage",
    "true_alpha",
    "harmful_fraction",
    "collapse_probability",
    "failure_probability",
    "system_failure",
    "failure_type",
    "oracle_action",
    "oracle_risk",
    "mechanism_label",
    "is_target",
    "ground_truth",
})


def assert_no_hidden_fields(value: Any, *, _path: tuple[str, ...] = ()) -> None:
    """Reject hidden metadata while allowing authentic public message text.

    Free-form public speech may legitimately contain words such as "failure".
    Leakage prevention therefore relies on a strict structured-field allowlist
    and provenance, rather than censoring the semantics of a public message.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(forbidden in lowered for forbidden in FORBIDDEN_FIELDS):
                raise ValueError(f"forbidden observation field: {key!r}")
            assert_no_hidden_fields(item, _path=(*_path, lowered))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            assert_no_hidden_fields(item, _path=_path)
        return
    is_public_message_text = bool(
        _path and _path[-1] == "text" and "messages" in _path
    )
    if isinstance(value, str) and not is_public_message_text:
        lowered = value.casefold()
        leaked = [field for field in FORBIDDEN_FIELDS if field in lowered]
        if leaked:
            raise ValueError(f"forbidden observation value contains: {sorted(leaked)}")


def observation_to_dict(observation: RegulatoryObservation) -> dict:
    payload = asdict(observation)
    if observation.messages is None:
        # Preserve the frozen v2 byte shape. Language-enabled observations use
        # an explicit tuple (possibly empty) and serialize the new field.
        payload.pop("messages")
    else:
        payload["schema_version"] = observation.schema_version
    payload["information_level"] = int(observation.information_level)
    assert_no_hidden_fields(payload)
    # Round-trip rejects unsupported values and normalizes tuples to JSON lists.
    return json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))


def observation_to_prompt(observation: RegulatoryObservation) -> str:
    prompt = json.dumps(
        observation_to_dict(observation),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return prompt


def observation_to_text(observation: RegulatoryObservation) -> str:
    """Render the public fields with a fixed, non-generative template."""
    market = observation.market
    social = observation.social
    lines = [
        f"Day {observation.day}.",
        f"Society size: {observation.society_size} agents.",
        "",
        "Market conditions:",
        f"- 1-day return: {market.return_1d:+.4f}",
        f"- 3-day return: {market.return_3d:+.4f}",
        f"- 3-day volatility: {market.volatility_3d:.4f}",
        f"- volume z-score: {market.volume_zscore:+.3f}",
        f"- liquidity: {market.liquidity:.4f}",
        f"- order imbalance: {market.order_imbalance:+.4f}",
        f"- cancellation rate: {market.cancellation_rate:.4f}",
        "",
        "Social activity:",
        f"- messages per agent: {social.message_rate:.4f}",
        f"- cascade reach: {social.cascade_reach:.4f}",
        f"- sentiment concentration: {social.sentiment_concentration:.4f}",
        f"- network degree concentration: {social.network_degree_concentration:.4f}",
        f"- information concentration: {social.information_concentration:.4f}",
        f"- consensus concentration: {social.consensus_concentration:.4f}",
    ]
    if observation.remaining_budget is not None:
        lines.insert(2, f"Remaining regulatory budget: {observation.remaining_budget:.2f}.")
    if observation.active_controls:
        lines.extend(["", "Active controls:"])
        for control in observation.active_controls:
            lines.append(f"- {control.action_type.upper()} on {control.target_id}")
    if observation.assets:
        lines.extend(["", "Observed assets:"])
        for asset in observation.assets:
            lines.extend([
                f"{asset.asset_id}:",
                f"- 3-day return: {asset.return_3d:+.4f}",
                f"- 3-day volatility: {asset.volatility_3d:.4f}",
                f"- volume z-score: {asset.volume_zscore:+.3f}",
                f"- order imbalance: {asset.order_imbalance:+.4f}",
                f"- cancellation rate: {asset.cancellation_rate:.4f}",
            ])
    if observation.entities:
        lines.extend(["", "Observed entities:"])
        for entity in observation.entities:
            lines.extend([
                f"{entity.entity_id}:",
                f"- message reach: {entity.message_reach:.4f}",
                f"- trading concentration: {entity.trading_concentration:.4f}",
                f"- cancellation rate: {entity.cancellation_rate:.4f}",
                f"- social influence: {entity.social_influence:.4f}",
                f"- activity change: {entity.activity_change:+.4f}",
            ])
    if observation.messages is not None:
        lines.extend(["", "Recent public messages:"])
        if not observation.messages:
            lines.append("- No public messages were observed in the recent window.")
        for message in observation.messages:
            lines.append(
                f"- Day {message.day}, {message.sender_id}, {message.asset_id} "
                f"({message.kind}, audience reach {message.audience_reach:.4f}): "
                f"{message.text}"
            )
    if observation.investigation is not None:
        investigation = observation.investigation
        lines.extend([
            "",
            f"Investigation result for {investigation.target_id}:",
            f"- target type: {investigation.target_type}",
            "- recent message pattern: "
            + ", ".join(f"{value:.4f}" for value in investigation.recent_message_pattern),
            "- recent order pattern: "
            + ", ".join(f"{value:.4f}" for value in investigation.recent_order_pattern),
            "- network neighbors: " + ", ".join(investigation.network_neighbors),
        ])
    if observation.history:
        lines.extend(["", "Interaction history:"])
        for item in observation.history:
            target = f" on {item.target_id}" if item.target_id else ""
            timing = "advanced one day" if item.advanced_day else "same-day evidence"
            lines.append(
                f"- #{item.interaction_index} day {item.day}: "
                f"{item.action_type.upper()}{target}; {timing}; cost {item.cost:.2f}"
            )
    # Validate the structured source before rendering. The fixed template adds
    # no simulator data of its own, while message text remains authentic public
    # speech rather than hidden metadata.
    observation_to_dict(observation)
    return "\n".join(lines)
