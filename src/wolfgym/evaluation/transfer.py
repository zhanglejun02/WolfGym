"""Paired external-rollout audit; never treats synthetic results as transfer."""
from __future__ import annotations

from itertools import combinations
from statistics import mean
from typing import Mapping, Sequence

from wolfgym.benchmark.management import strict_label
from wolfgym.evaluation.management import action_label


def compare_transfer(reference: Sequence[Mapping], external: Sequence[Mapping], *,
                     backend_revision: str, effect_margin: float = 0.05) -> dict:
    """Compare real saved continuations from the same source snapshots.

    External rows must carry source_instance_id, source_trajectory_id and
    backend_revision in provenance. Public IDs and candidate costs must agree.
    This audits submitted evidence; it does not launch or certify a provider.
    """
    if not backend_revision.strip() or effect_margin < 0:
        raise ValueError("backend_revision and a non-negative effect margin are required")
    original = {row["instance_id"]: row for row in reference}
    if len(original) != len(reference):
        raise ValueError("duplicate reference states")
    paired = []
    seen = set()
    for row in external:
        provenance = row["provenance"]
        source = provenance["source_instance_id"]
        if source in seen or source not in original:
            raise ValueError("external states must uniquely match reference states")
        seen.add(source)
        base = original[source]
        if provenance.get("backend_revision") != backend_revision:
            raise ValueError("external backend revision does not match")
        if provenance.get("source_trajectory_id") != base["trajectory_id"]:
            raise ValueError("external source trajectory does not match")
        if row["counterfactuals"]["horizon"] != base["counterfactuals"]["horizon"]:
            raise ValueError("transfer horizons must match")
        left = base["counterfactuals"]["actions"]
        right = row["counterfactuals"]["actions"]
        if set(left) != set(right) or any(
            left[key]["action"] != right[key]["action"] or left[key]["cost"] != right[key]["cost"]
            for key in left
        ):
            raise ValueError("transfer candidates, targets and costs must match")
        for record in (base, row):
            seeds = record["counterfactuals"]["seeds"]
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("each backend requires unique continuation seeds")
            for outcome in record["counterfactuals"]["actions"].values():
                trials = outcome["failure_outcomes"]
                if len(trials) != len(seeds) or any(type(value) is not bool for value in trials):
                    raise ValueError("transfer requires executed per-seed outcomes")
                if abs(mean(trials) - outcome["failure_probability"]) > 1e-12:
                    raise ValueError("transfer probability disagrees with executed outcomes")
        a, b = strict_label(base), strict_label(row)
        pa, pb = left["WAIT"]["failure_probability"], right["WAIT"]["failure_probability"]
        sign = lambda delta: 1 if delta > effect_margin else -1 if delta < -effect_margin else 0
        paired.append({
            "source_instance_id": source, "reference_risk": pa, "external_risk": pb,
            "effect_agreement": mean(
                sign(pa - left[key]["failure_probability"]) == sign(pb - right[key]["failure_probability"])
                for key in left if key != "WAIT"
            ) if len(left) > 1 else None,
            "feasibility_agreement": a["feasible"] == b["feasible"],
            "both_feasible": a["feasible"] and b["feasible"],
            "valid_set_overlap": bool(
                {action_label(item) for item in a["valid_actions"]}
                & {action_label(item) for item in b["valid_actions"]}
            ),
        })
    ordered_pairs = [(a, b) for a, b in combinations(paired, 2)
                     if a["reference_risk"] != b["reference_risk"]]
    feasible_pairs = [row for row in paired if row["both_feasible"]]
    effects = [row["effect_agreement"] for row in paired if row["effect_agreement"] is not None]
    return {
        "backend_revision": backend_revision, "paired_states": len(paired),
        "reference_states": len(reference), "effect_direction_agreement": mean(effects) if effects else None,
        "baseline_risk_order_agreement": mean(
            (a["reference_risk"] - b["reference_risk"]) * (a["external_risk"] - b["external_risk"]) > 0
            for a, b in ordered_pairs
        ) if ordered_pairs else None,
        "ordered_pair_count": len(ordered_pairs),
        "feasibility_agreement": mean(row["feasibility_agreement"] for row in paired) if paired else None,
        "IS_valid_set_overlap": mean(row["valid_set_overlap"] for row in feasible_pairs) if feasible_pairs else None,
        "IS_denominator": len(feasible_pairs), "pairs": paired,
        "controller_transfer_evidence": "Run matched CR episodes on both populations; static agreement alone does not establish controller transfer.",
    }
